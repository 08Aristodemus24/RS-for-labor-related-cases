import os.path
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
from dataset import get_tg_dataset
from model import *
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from utils import *

np.random.seed(123)
np.random.seed()

dataset_name = "clean_more_than_subject_person_subjectdup"
dataset_duplicates = dataset_name + ".map"
selected_features = ["birthYear", "birthDate", "birthPlace", "deathYear", "deathDate",
                     "activeYearsStartYear", "deathPlace", "activeYearsEndYear",
                     "almaMater", "deathCause", "restingPlace", "education",
                     "residence", "religion", "nationality", "stateOfOrigin",
                     "knownFor", "party", "ethnicity", "award", "networth", "hometown",
                     "employer", "board", "citizenship"]

cuda = 1
batch_size = 8
epoch_log = 10
epoch_num = 2001
comment = "task64"

# set up torch device (default: gpu)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

device = torch.device('cuda:' + str(cuda))

data_list = get_tg_dataset(dataset_name, dataset_duplicates, selected_features=selected_features)

writer_train = SummaryWriter(comment=dataset_name + "_" + comment + '_train')

# initialize the model
feature_shapes = data_list[0].feature_shapes
output_dim = 8  # 2
hidden_dim = 16  # 4

model = locals()["Baseline"](feature_shapes=feature_shapes, hidden_dim=hidden_dim,
                             output_dim=output_dim, device=device).to(device)

linear = locals()["Linear"](input_dim=len(feature_shapes), output_dim=1).to(device)

for i, data in enumerate(data_list):
    prepared_features = []

    index_shift = 0

    for j in range(len(feature_shapes)):
        start_index = index_shift
        end_index = start_index + feature_shapes[j]
        index_shift = end_index

        prepared_features.append(torch.index_select(data.x, 1, torch.from_numpy(
            np.asarray(range(data.x.shape[1]))[start_index:end_index]).long()))

    data.prepared_features = prepared_features
    data = data.to(device)
    data_list[i] = data

# initialize optimizer
params = list(model.parameters()) + list(linear.parameters())
optimizer = torch.optim.Adam(params, lr=1e-2, weight_decay=5e-4)

# initialize loss and cosine function
loss_func = nn.BCEWithLogitsLoss()
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
euclidean = nn.PairwiseDistance()

max_val_score = 0

for epoch in range(epoch_num):
    if epoch == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10

    model.train()
    linear.train()
    shuffle(data_list)
    effective_len = len(data_list) // batch_size * len(data_list)

    # iterate over all subgraphs
    for id, data in enumerate(data_list[:effective_len]):
        # resample negative duplicate sets
        get_duplicate_mask(data, resplit=False, use_jaccard_ranking=False)

        # skip subgraphs without positive samples
        if data.mask_duplicate_positive_train.shape[1] == 0:
            continue

        duplicate_mask_train = np.concatenate(
            (data.mask_duplicate_positive_train, data.mask_duplicate_negative_train),
            axis=-1)

        # prepare labels based on the shape of positive and negative sample sets
        label_positive = torch.ones([data.mask_duplicate_positive_train.shape[1], ], dtype=np.float)
        label_negative = torch.zeros([data.mask_duplicate_negative_train.shape[1], ], dtype=np.float)

        label = torch.cat((label_positive, label_negative)).to(device)

        feats_first = []
        feats_second = []
        for j in range(len(feature_shapes)):
            feats_first.append(
                torch.index_select(data.prepared_features[j], 0, torch.from_numpy(duplicate_mask_train[0, :]).long()))
            feats_second.append(
                torch.index_select(data.prepared_features[j], 0, torch.from_numpy(duplicate_mask_train[1, :]).long()))

        nodes_first_out = model(feats_first)
        nodes_second_out = model(feats_second)

        predictions = []
        for j in range(len(feature_shapes)):
            # pred = torch.sum(nodes_first_out[j] * nodes_second_out[j], dim=-1)
            # pred = cos(nodes_first_out[j], nodes_second_out[j])
            pred = euclidean(nodes_first_out[j], nodes_second_out[j])
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=-1)
        predictions = linear(predictions).double().to(device)
        label = torch.reshape(label, (label.shape[0], 1))

        # calculate the loss for the predictions compared to the labels
        loss = loss_func(predictions, label)

        loss.backward()
        # update
        if id % batch_size == batch_size - 1:
            if batch_size > 1:
                # if this is slow, no need to do this normalization
                for p in params:
                    if p.grad is not None:
                        p.grad /= batch_size
            optimizer.step()
            optimizer.zero_grad()

    # save model to path after each epoch
    model_path = "./models/" + dataset_name + "_" + str(comment) + ".pt"

    # start evaluation after defined epoch count
    if epoch % epoch_log == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': linear.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_path + "-linear")

        model.eval()
        linear.eval()

        total_labels_train = np.asarray([])
        total_labels_test = np.asarray([])
        total_labels_val = np.asarray([])

        total_predictions_train = np.asarray([])
        total_predictions_test = np.asarray([])
        total_predictions_val = np.asarray([])

        total_loss_train = np.asarray([])
        total_loss_test = np.asarray([])
        total_loss_val = np.asarray([])

        # iterate over all subgraphs
        for id, data in enumerate(data_list[:effective_len]):
            # resample negative duplicate sets
            get_duplicate_mask(data, resplit=False, use_jaccard_ranking=False)

            # skip subgraphs without positive samples
            if data.mask_duplicate_positive_train.shape[1] == 0:
                continue

            # training
            duplicate_mask_train = np.concatenate(
                (data.mask_duplicate_positive_train, data.mask_duplicate_negative_train),
                axis=-1)

            # prepare labels based on the shape of positive and negative sample sets
            label_positive = torch.ones([data.mask_duplicate_positive_train.shape[1], ], dtype=np.float)
            label_negative = torch.zeros([data.mask_duplicate_negative_train.shape[1], ], dtype=np.float)

            label = torch.cat((label_positive, label_negative)).to(device)

            feats_first = []
            feats_second = []
            for j in range(len(feature_shapes)):
                feats_first.append(torch.index_select(data.prepared_features[j], 0,
                                                      torch.from_numpy(duplicate_mask_train[0, :]).long()))
                feats_second.append(torch.index_select(data.prepared_features[j], 0,
                                                       torch.from_numpy(duplicate_mask_train[1, :]).long()))

            nodes_first_out = model(feats_first)
            nodes_second_out = model(feats_second)

            predictions = []
            for j in range(len(feature_shapes)):
                # pred = torch.sum(nodes_first_out[j] * nodes_second_out[j], dim=-1)
                # pred = cos(nodes_first_out[j], nodes_second_out[j])
                pred = euclidean(nodes_first_out[j], nodes_second_out[j])
                predictions.append(pred)

            predictions = torch.stack(predictions, dim=-1)
            predictions = linear(predictions).double().to(device)
            label = torch.reshape(label, (label.shape[0], 1))

            # calculate the loss for the predictions compared to the labels
            loss = loss_func(predictions, label)

            total_labels_train = np.append(total_labels_train, label.flatten().cpu().numpy())
            total_predictions_train = np.append(total_predictions_train, predictions.flatten().data.cpu().numpy())
            total_loss_train = np.append(total_loss_train, loss.item())

            get_duplicate_mask(data, resplit=False, use_jaccard_ranking=False)

            # testing
            duplicate_mask_test = np.concatenate(
                (data.mask_duplicate_positive_test, data.mask_duplicate_negative_test),
                axis=-1)

            # prepare labels based on the shape of positive and negative sample sets
            label_positive = torch.ones([data.mask_duplicate_positive_test.shape[1], ], dtype=np.float)
            label_negative = torch.zeros([data.mask_duplicate_negative_test.shape[1], ], dtype=np.float)

            label = torch.cat((label_positive, label_negative)).to(device)

            feats_first = []
            feats_second = []
            for j in range(len(feature_shapes)):
                feats_first.append(torch.index_select(data.prepared_features[j], 0,
                                                      torch.from_numpy(duplicate_mask_test[0, :]).long()))
                feats_second.append(torch.index_select(data.prepared_features[j], 0,
                                                       torch.from_numpy(duplicate_mask_test[1, :]).long()))

            nodes_first_out = model(feats_first)
            nodes_second_out = model(feats_second)

            predictions = []
            for j in range(len(feature_shapes)):
                # pred = torch.sum(nodes_first_out[j] * nodes_second_out[j], dim=-1)
                # pred = cos(nodes_first_out[j], nodes_second_out[j])
                pred = euclidean(nodes_first_out[j], nodes_second_out[j])
                predictions.append(pred)

            predictions = torch.stack(predictions, dim=-1)
            predictions = linear(predictions).double().to(device)
            label = torch.reshape(label, (label.shape[0], 1))

            # calculate the loss for the predictions compared to the labels
            loss = loss_func(predictions, label)

            total_labels_test = np.append(total_labels_test, label.flatten().cpu().numpy())
            total_predictions_test = np.append(total_predictions_test, predictions.flatten().data.cpu().numpy())
            total_loss_test = np.append(total_loss_test, loss.item())

            get_duplicate_mask(data, resplit=False, use_jaccard_ranking=False)
            # validation
            duplicate_mask_val = np.concatenate(
                (data.mask_duplicate_positive_val, data.mask_duplicate_negative_val),
                axis=-1)

            # prepare labels based on the shape of positive and negative sample sets
            label_positive = torch.ones([data.mask_duplicate_positive_val.shape[1], ], dtype=np.float)
            label_negative = torch.zeros([data.mask_duplicate_negative_val.shape[1], ], dtype=np.float)

            label = torch.cat((label_positive, label_negative)).to(device)

            feats_first = []
            feats_second = []
            for j in range(len(feature_shapes)):
                feats_first.append(torch.index_select(data.prepared_features[j], 0,
                                                      torch.from_numpy(duplicate_mask_val[0, :]).long()))
                feats_second.append(torch.index_select(data.prepared_features[j], 0,
                                                       torch.from_numpy(duplicate_mask_val[1, :]).long()))

            nodes_first_out = model(feats_first)
            nodes_second_out = model(feats_second)

            predictions = []
            for j in range(len(feature_shapes)):
                # pred = torch.sum(nodes_first_out[j] * nodes_second_out[j], dim=-1)
                # pred = cos(nodes_first_out[j], nodes_second_out[j])
                pred = euclidean(nodes_first_out[j], nodes_second_out[j])
                predictions.append(pred)

            predictions = torch.stack(predictions, dim=-1)
            predictions = linear(predictions).double().to(device)
            label = torch.reshape(label, (label.shape[0], 1))

            # calculate the loss for the predictions compared to the labels
            loss = loss_func(predictions, label)

            total_labels_val = np.append(total_labels_val, label.flatten().cpu().numpy())
            total_predictions_val = np.append(total_predictions_val, predictions.flatten().data.cpu().numpy())
            total_loss_val = np.append(total_loss_val, loss.item())

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
            torch.save({
                'epoch': epoch,
                'model_state_dict': linear.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, model_path + "-max-linear")

        print(epoch, 'Train AUC: {:.4f}'.format(train_auc_score), 'Train Loss: {:.4f}'.format(train_loss_score),
              '  Test AUC: {:.4f}'.format(test_auc_score), 'Test Loss: {:.4f}'.format(test_loss_score),
              '  Val AUC: {:.4f}'.format(val_auc_score), 'Val Loss: {:.4f}'.format(val_loss_score))

        writer_train.add_scalar('/auc_' + dataset_name, test_auc_score, epoch)
        writer_train.add_scalar('/loss_' + dataset_name, test_loss_score, epoch)

print("Finished.")
