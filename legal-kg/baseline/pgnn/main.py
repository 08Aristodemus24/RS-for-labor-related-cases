import os.path
import time
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
from args import make_args
from dataset import get_tg_dataset
from model import *
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from utils import preselect_anchor, get_duplicate_mask

# nohup python main.py --model PGNN --layer_num 2 --epoch_num 2001 --dataset duplicate_without_subject_feat_25percent_missing_edges_udbms --dataset_duplicates person_duplicate_without_subject_feat_25percent.map &
# nohup python main.py --model PGNN --layer_num 2 --epoch_num 2001 --dataset clean_more_than_subject_person_dup25 --dataset_duplicates clean_more_than_subject_person_dup25.map &

# COMMAND: python main.py --model PGNN --layer_num 2 --dataset duplicate_without_subject_feat_25percent_missing_edges_udbms
# COMMAND: nohup python main.py --model PGNN_plus --layer_num 2 --epoch_num 4001 --dataset clean_more_than_subject_person_subjectdup --dataset_duplicates clean_more_than_subject_person_subjectdup.map --comment {TASK_ID} &

if not os.path.isdir('results'):
    os.mkdir('results')

# parse args and print them
args = make_args()
print(args)

np.random.seed(123)
np.random.seed()
start_time = time.time()

# initialize tensorboardX SummaryWriters with unique name
writer_train = SummaryWriter(
    comment=args.dataset + "_" + args.model + "_" + args.task + "_" + str(args.layer_num) + "layer_" + str(
        start_time) + '_' + args.comment + '_train')
writer_val = SummaryWriter(
    comment=args.dataset + "_" + args.model + "_" + args.task + "_" + str(args.layer_num) + "layer_" + str(
        start_time) + '_' + args.comment + '_val')
writer_test = SummaryWriter(
    comment=args.dataset + "_" + args.model + "_" + args.task + "_" + str(args.layer_num) + "layer_" + str(
        start_time) + '_' + args.comment + '_test')

# set up torch device (default: gpu)
if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print('Using CPU')
device = torch.device('cuda:' + str(args.cuda) if args.gpu else 'cpu')

datasets_name = [args.dataset]

for dataset_name in datasets_name:
    results = []

    for repeat in range(args.repeat_num):
        result_val = []
        result_test = []

        time1 = time.time()
        data_list = get_tg_dataset(args, dataset_name, use_cache=args.cache, remove_feature=args.rm_feature,
                                   selected_features=["birthYear", "birthDate", "birthPlace", "deathYear", "deathDate",
                                                      "activeYearsStartYear", "deathPlace", "activeYearsEndYear",
                                                      "almaMater", "deathCause", "restingPlace", "education",
                                                      "residence", "religion", "nationality", "stateOfOrigin",
                                                      "knownFor", "party", "ethnicity", "award", "networth", "hometown",
                                                      "employer", "board", "citizenship"]
                                   )

        time2 = time.time()
        print(dataset_name, 'load time', time2 - time1)

        num_features = data_list[0].x.shape[1]
        print("Number of features: " + str(num_features))
        num_node_classes = None
        num_graph_classes = None
        if 'y' in data_list[0].__dict__ and data_list[0].y is not None:
            num_node_classes = max([data.y.max().item() for data in data_list]) + 1
        if 'y_graph' in data_list[0].__dict__ and data_list[0].y_graph is not None:
            num_graph_classes = max([data.y_graph.numpy()[0] for data in data_list]) + 1
        print('Dataset', dataset_name, 'Graph', len(data_list), 'Feature', num_features, 'Node Class',
              num_node_classes, 'Graph Class', num_graph_classes)
        nodes = [data.num_nodes for data in data_list]
        print("Nodes:")
        print(nodes)
        edges = [data.num_edges for data in data_list]
        print("Edges:")
        print(edges)
        print('Node: max{}, min{}, mean{}'.format(max(nodes), min(nodes), sum(nodes) / len(nodes)))
        print('Edge: max{}, min{}, mean{}'.format(max(edges), min(edges), sum(edges) / len(edges)))

        args.batch_size = min(args.batch_size, len(data_list))
        print('Anchor num {}, Batch size {}'.format(args.anchor_num, args.batch_size))

        # prepare data and preselect anchors randomly
        for i, data in enumerate(data_list):
            preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')

            data = data.to(device)
            data_list[i] = data

        # initialize the model
        input_dim = num_features
        output_dim = args.output_dim
        model = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                                     hidden_dim=args.hidden_dim, output_dim=output_dim,
                                     feature_pre=args.feature_pre, layer_num=args.layer_num,
                                     dropout=args.dropout).to(device)
        # initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

        # initialize loss function and sigmoid
        loss_func = nn.BCEWithLogitsLoss()
        out_act = nn.Sigmoid()
        euclidean = nn.PairwiseDistance()

        max_val_score = 0

        # start training process in for loop for defined epoch length
        for epoch in range(args.epoch_num):
            if epoch == 200:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10

            # set model to training mode
            model.train()
            optimizer.zero_grad()

            # shuffle objects in data_list randomly
            shuffle(data_list)
            effective_len = len(data_list) // args.batch_size * len(data_list)

            # iterate over all subgraphs
            for id, data in enumerate(data_list[:effective_len]):
                # skip subgraphs without positive samples
                if data.mask_duplicate_positive_train.shape[1] == 0:
                    continue

                # preselect anchors again if permute arg is enabled
                if args.permute:
                    preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)

                # get node embeddings of each node in subgraph by using current model state
                out = model(data)

                # resample negative duplicate sets
                get_duplicate_mask(data, resplit=False, use_jaccard_ranking=False)

                # prepare mask containing n positive samples and m negative samples => (..., ..., n, ..., ..., m)
                duplicate_mask_train = np.concatenate(
                    (data.mask_duplicate_positive_train, data.mask_duplicate_negative_train),
                    axis=-1)

                nodes_first = torch.index_select(out, 0,
                                                 torch.from_numpy(duplicate_mask_train[0, :]).long().to(device))

                nodes_second = torch.index_select(out, 0,
                                                  torch.from_numpy(duplicate_mask_train[1, :]).long().to(device))

                # calculate prediction by using the dot product of both vectors
                pred = torch.sum(nodes_first * nodes_second, dim=-1)
                # pred = euclidean(nodes_first, nodes_second)

                # prepare labels based on the shape of positive and negative sample sets
                label_positive = torch.ones([data.mask_duplicate_positive_train.shape[1], ], dtype=pred.dtype)
                label_negative = torch.zeros([data.mask_duplicate_negative_train.shape[1], ], dtype=pred.dtype)

                label = torch.cat((label_positive, label_negative)).to(device)

                # calculate the loss for the predictions compared to the labels
                loss = loss_func(pred, label)

                # update
                loss.backward()
                if id % args.batch_size == args.batch_size - 1:
                    if args.batch_size > 1:
                        # if this is slow, no need to do this normalization
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad /= args.batch_size
                    optimizer.step()
                    optimizer.zero_grad()

            # save model to path after each epoch
            model_path = "./models/" + args.dataset + "_" + args.model + "_" + str(args.epoch_num) + "_" + args.task + \
                         "_" + str(args.layer_num) + "layer" + str(start_time) + "_task" + str(args.comment) + ".pt"

            # start evaluation after defined epoch count
            if epoch % args.epoch_log == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, model_path)

                # set model to evaluation mode
                model.eval()

                total_labels_train = np.asarray([])
                total_labels_test = np.asarray([])
                total_labels_val = np.asarray([])

                total_predictions_train = np.asarray([])
                total_predictions_test = np.asarray([])
                total_predictions_val = np.asarray([])

                total_loss_train = np.asarray([])
                total_loss_test = np.asarray([])
                total_loss_val = np.asarray([])

                for id, data in enumerate(data_list):
                    # skip subgraphs without positive samples
                    if data.mask_duplicate_positive_train.shape[1] == 0:
                        continue

                    out = model(data)

                    # train
                    get_duplicate_mask(data, resplit=False, use_jaccard_ranking=False)
                    duplicate_mask_train = np.concatenate(
                        (data.mask_duplicate_positive_train, data.mask_duplicate_negative_train), axis=-1)

                    nodes_first = torch.index_select(out, 0,
                                                     torch.from_numpy(duplicate_mask_train[0, :]).long().to(device))

                    nodes_second = torch.index_select(out, 0,
                                                      torch.from_numpy(duplicate_mask_train[1, :]).long().to(
                                                          device))

                    pred = torch.sum(nodes_first * nodes_second, dim=-1)
                    # pred = euclidean(nodes_first, nodes_second)

                    label_positive = torch.ones([data.mask_duplicate_positive_train.shape[1], ], dtype=pred.dtype)
                    label_negative = torch.zeros([data.mask_duplicate_negative_train.shape[1], ], dtype=pred.dtype)
                    label = torch.cat((label_positive, label_negative)).to(device)

                    total_labels_train = np.append(total_labels_train, label.flatten().cpu().numpy())
                    total_predictions_train = np.append(total_predictions_train,
                                                        pred.flatten().data.cpu().numpy())
                    total_loss_train = np.append(total_loss_train, loss.item())

                    # val
                    get_duplicate_mask(data, resplit=False)
                    duplicate_mask_val = np.concatenate(
                        (data.mask_duplicate_positive_val, data.mask_duplicate_negative_val), axis=-1)

                    nodes_first = torch.index_select(out, 0,
                                                     torch.from_numpy(duplicate_mask_val[0, :]).long().to(device))

                    nodes_second = torch.index_select(out, 0,
                                                      torch.from_numpy(duplicate_mask_val[1, :]).long().to(
                                                          device))

                    pred = torch.sum(nodes_first * nodes_second, dim=-1)
                    # pred = euclidean(nodes_first, nodes_second)

                    label_positive = torch.ones([data.mask_duplicate_positive_val.shape[1], ], dtype=pred.dtype)
                    label_negative = torch.zeros([data.mask_duplicate_negative_val.shape[1], ], dtype=pred.dtype)
                    label = torch.cat((label_positive, label_negative)).to(device)

                    total_labels_val = np.append(total_labels_val, label.flatten().cpu().numpy())
                    total_predictions_val = np.append(total_predictions_val, pred.flatten().data.cpu().numpy())
                    total_loss_val = np.append(total_loss_val, loss.item())

                    # test
                    get_duplicate_mask(data, resplit=False)
                    duplicate_mask_test = np.concatenate(
                        (data.mask_duplicate_positive_test, data.mask_duplicate_negative_test), axis=-1)

                    nodes_first = torch.index_select(out, 0,
                                                     torch.from_numpy(duplicate_mask_test[0, :]).long().to(device))

                    nodes_second = torch.index_select(out, 0,
                                                      torch.from_numpy(duplicate_mask_test[1, :]).long().to(
                                                          device))

                    pred = torch.sum(nodes_first * nodes_second, dim=-1)
                    # pred = euclidean(nodes_first, nodes_second)

                    label_positive = torch.ones([data.mask_duplicate_positive_test.shape[1], ], dtype=pred.dtype)
                    label_negative = torch.zeros([data.mask_duplicate_negative_test.shape[1], ], dtype=pred.dtype)
                    label = torch.cat((label_positive, label_negative)).to(device)

                    total_labels_test = np.append(total_labels_test, label.flatten().cpu().numpy())
                    total_predictions_test = np.append(total_predictions_test, pred.flatten().data.cpu().numpy())
                    total_loss_test = np.append(total_loss_test, loss.item())

                train_auc_score = roc_auc_score(total_labels_train, total_predictions_train)
                train_loss_score = np.average(total_loss_train)

                test_auc_score = roc_auc_score(total_labels_test, total_predictions_test)
                test_loss_score = np.average(total_loss_test)

                val_auc_score = roc_auc_score(total_labels_val, total_predictions_val)
                val_loss_score = np.average(total_loss_val)

                if val_auc_score > max_val_score:
                    max_val_score = val_auc_score

                    print("=> model saved, Val AUC:", val_auc_score)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, model_path + "-max")

                print(repeat, epoch, 'Loss {:.4f}'.format(train_loss_score),
                      'Train AUC: {:.4f}'.format(train_auc_score),
                      'Val AUC: {:.4f}'.format(val_auc_score), 'Test AUC: {:.4f}'.format(test_auc_score))

                writer_train.add_scalar('repeat_' + str(repeat) + '/auc_' + dataset_name, train_auc_score, epoch)
                writer_train.add_scalar('repeat_' + str(repeat) + '/loss_' + dataset_name, train_loss_score, epoch)
                writer_val.add_scalar('repeat_' + str(repeat) + '/auc_' + dataset_name, val_auc_score, epoch)
                writer_train.add_scalar('repeat_' + str(repeat) + '/loss_' + dataset_name, val_loss_score, epoch)
                writer_test.add_scalar('repeat_' + str(repeat) + '/auc_' + dataset_name, test_auc_score, epoch)
                writer_test.add_scalar('repeat_' + str(repeat) + '/loss_' + dataset_name, test_loss_score, epoch)
                result_val.append(val_auc_score)
                result_test.append(test_auc_score)

        result_val = np.array(result_val)
        result_test = np.array(result_test)
        results.append(result_test[np.argmax(result_val)])
    results = np.array(results)
    results_mean = np.mean(results).round(6)
    results_std = np.std(results).round(6)
    print('-----------------Final-------------------')
    print(results_mean, results_std)
    with open(
            'results/{}_{}_{}_layer{}_{}_approximate{}.txt'.format(args.task, args.model, dataset_name, args.layer_num,
                                                                   str(start_time), args.approximate), 'w') as f:
        f.write('{}, {}\n'.format(results_mean, results_std))

# export scalar data to JSON for external processing
writer_train.export_scalars_to_json(
    "./runs/" + args.dataset + "_" + args.model + "_" + args.task + "_" + str(args.layer_num) + "layer_" + str(
        start_time) + "_all_scalars.json")
writer_train.close()
writer_val.export_scalars_to_json(
    "./runs/" + args.dataset + "_" + args.model + "_" + args.task + "_" + str(args.layer_num) + "layer_" + str(
        start_time) + "_all_scalars.json")
writer_val.close()
writer_test.export_scalars_to_json(
    "./runs/" + args.dataset + "_" + args.model + "_" + args.task + "_" + str(args.layer_num) + "layer_" + str(
        start_time) + "_all_scalars.json")
writer_test.close()
