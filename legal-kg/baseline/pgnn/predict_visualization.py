from sklearn.metrics import roc_auc_score, accuracy_score

from args import *
from model import *
from utils import *
from dataset import *

import pickle
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

args = make_args()
print(args)

# set up gpu
if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print('Using CPU')
device = torch.device('cuda:' + str(args.cuda) if args.gpu else 'cpu')

##
model_name = "duplicate_without_subject_feat_25percent_missing_edges_udbms_PGNN_similarity_2layer1568415280.3627334.pt"
dir_name = model_name.replace(".", "+")
dataset_name = "clean_more_than_subject_person"
prefix = ""
##

if not os.path.exists("./plots/" + dir_name):
    os.makedirs("./plots/" + dir_name)

data_list = get_tg_dataset(args, dataset_name, use_cache=True, remove_feature=args.rm_feature)

model = locals()[args.model](input_dim=data_list[0].x.shape[1], feature_dim=args.feature_dim,
                             hidden_dim=args.hidden_dim, output_dim=args.output_dim,
                             feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)

num_features = data_list[0].x.shape[1]
print("Number of features: " + str(num_features))
num_node_classes = None
num_graph_classes = None
if 'y' in data_list[0].__dict__ and data_list[0].y is not None:
    num_node_classes = max([data.y.max().item() for data in data_list]) + 1
if 'y_graph' in data_list[0].__dict__ and data_list[0].y_graph is not None:
    num_graph_classes = max([data.y_graph.numpy()[0] for data in data_list]) + 1
print('Dataset', dataset_name, 'Graph', len(data_list), 'Feature', num_features, 'Node Class', num_node_classes,
      'Graph Class', num_graph_classes)

nodes = [data.num_nodes for data in data_list]
edges = [data.num_edges for data in data_list]
print('Node: max{}, min{}, mean{}'.format(max(nodes), min(nodes), sum(nodes) / len(nodes)))
print('Edge: max{}, min{}, mean{}'.format(max(edges), min(edges), sum(edges) / len(edges)))

args.batch_size = min(args.batch_size, len(data_list))
print('Anchor num {}, Batch size {}'.format(args.anchor_num, args.batch_size))

# data
for i, data in enumerate(data_list):
    preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')
    data = data.to(device)
    data_list[i] = data

# model
input_dim = num_features
output_dim = args.output_dim
model = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                             hidden_dim=args.hidden_dim, output_dim=output_dim,
                             feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)
# loss
loss_func = nn.BCEWithLogitsLoss()
out_act = nn.Sigmoid()

# load the checkpoint
checkpoint = torch.load("./models/" + model_name, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

epoch = checkpoint['epoch']
loss = checkpoint['loss']

pos_threshold = 0.9
pos_predictions_over_threshold = []
pos_predictions_under_threshold = []

neg_threshold = 0.3
neg_predictions_over_threshold = []
neg_predictions_under_threshold = []

for id, data in enumerate(data_list):
    embeddings = model(data)
    sub_graph = data.graph

    print("Subgraph " + str(id) + ":")

    print("======================================= Duplicate Mapping ===")
    print("length: " + str(len(data.graph_duplicate_node_mappings)))
    print(data.graph_duplicate_node_mappings)
    print("")

    pos_predictions = []
    neg_predictions = []

    mask_duplicate = np.concatenate((data.mask_duplicate_positive_train,
                                     data.mask_duplicate_positive_test,
                                     data.mask_duplicate_positive_val),
                                    axis=1)

    neg_samples = list(get_mask_duplicate_negative(mask_duplicate,
                                                   num_nodes=data.num_nodes,
                                                   num_negtive_duplicates=mask_duplicate.shape[1]))

    for tup in data.graph_duplicate_node_mappings:
        node_first = torch.index_select(embeddings, 0, torch.tensor([tup[0]]).long().to(device))
        node_second = torch.index_select(embeddings, 0, torch.tensor([tup[1]]).long().to(device))

        pred = torch.sum(node_first * node_second, dim=-1)

        prediction = float(pred.data[0])

        if prediction > pos_threshold:
            pos_predictions_over_threshold.append((id, tup[0], tup[1], prediction))
            pos_predictions.append((True, str(tup[0]) + " <=> " + str(tup[1]) + " : " + str(prediction)))
        else:
            pos_predictions_under_threshold.append((id, tup[0], tup[1], prediction))
            pos_predictions.append((False, str(tup[0]) + " <=> " + str(tup[1]) + " : " + str(prediction)))

    for i in range(len(neg_samples[0])):
        node_first = torch.index_select(embeddings, 0, torch.tensor([neg_samples[0][i]]).long().to(device))
        node_second = torch.index_select(embeddings, 0, torch.tensor([neg_samples[1][i]]).long().to(device))

        pred = torch.sum(node_first * node_second, dim=-1)

        prediction = float(pred.data[0])

        if prediction < neg_threshold:
            neg_predictions_over_threshold.append((id, neg_samples[0][i], neg_samples[1][i], prediction))
            neg_predictions.append((True, str(neg_samples[0][i]) + " <=> " + str(neg_samples[1][i]) + " : " + str(prediction)))
        else:
            neg_predictions_under_threshold.append((id, neg_samples[0][i], neg_samples[1][i], prediction))
            neg_predictions.append((False, str(neg_samples[0][i]) + " <=> " + str(neg_samples[1][i]) + " : " + str(prediction)))

    mapping = {y: str(y) + ": " + str(x) for x, y in data.mapping.items()}

    nx.relabel_nodes(sub_graph, mapping, copy=False)

    fig, ax = plt.subplots(2)

    color_map = []
    for node in sub_graph:
        if node.endswith('_(duplicate)'):
            color_map.append('red')
        else:
            color_map.append('blue')

    nx.draw(sub_graph, with_labels=True, node_color=color_map, font_size=5, ax=ax[0],
            pos=nx.kamada_kawai_layout(sub_graph))

    ax[1].set_title("Subgraph %s" % str(id))
    ax[1].axis('off')
    ax[1].text(0, 0.94, "Duplicate Mappings:", fontsize=8)
    ax[1].text(0.45, 0.94, "Negative Mappings:", fontsize=8)

    y = 0.85
    for pred in pos_predictions:
        ax[1].text(0, y, pred[1], fontsize=6, color='green' if pred[0] == True else 'red')
        y = y - 0.1

    y = 0.85
    for pred in neg_predictions:
        ax[1].text(0.45, y, pred[1], fontsize=6, color='green' if pred[0] == True else 'red')
        y = y - 0.1

    plt.savefig("./plots/" + dir_name + "/" + dataset_name + prefix + "-subgraph" + str(id) + ".png",
                bbox_inches="tight", dpi=600)

with open("./plots/" + dir_name + "/" + dataset_name + prefix + "_pos_over_threshold.dat", 'wb') as f1, \
        open("./plots/" + dir_name + "/" + dataset_name + prefix + "_pos_under_threshold.dat", 'wb') as f2, \
        open("./plots/" + dir_name + "/" + dataset_name + prefix + "_neg_over_threshold.dat", 'wb') as f3, \
        open("./plots/" + dir_name + "/" + dataset_name + prefix + "_neg_under_threshold.dat", 'wb') as f4:
    pickle.dump(pos_predictions_over_threshold, f1, protocol=4)
    pickle.dump(pos_predictions_under_threshold, f2, protocol=4)
    pickle.dump(neg_predictions_over_threshold, f3, protocol=4)
    pickle.dump(neg_predictions_over_threshold, f4, protocol=4)

pos_predictions_over_threshold_count = len(pos_predictions_over_threshold)
pos_predictions_under_threshold_count = len(pos_predictions_under_threshold)
pos_predictions_count = pos_predictions_over_threshold_count + pos_predictions_under_threshold_count

neg_predictions_over_threshold_count = len(neg_predictions_over_threshold)
neg_predictions_under_threshold_count = len(neg_predictions_under_threshold)
neg_predictions_count = neg_predictions_over_threshold_count + neg_predictions_under_threshold_count

print("positive predictions over threshold of %s: %s (%.2f%%)" % (
    pos_threshold, pos_predictions_over_threshold_count, pos_predictions_over_threshold_count / pos_predictions_count * 100))

print("positive predictions under threshold of %s: %s (%.2f%%)" % (
    pos_threshold, pos_predictions_under_threshold_count, pos_predictions_under_threshold_count / pos_predictions_count * 100))

print("total positive prediction count: %s" % pos_predictions_count)


print("negative predictions over threshold of %s: %s (%.2f%%)" % (
    neg_threshold, neg_predictions_over_threshold_count, neg_predictions_over_threshold_count / neg_predictions_count * 100))

print("negative predictions under threshold of %s: %s (%.2f%%)" % (
    neg_threshold, neg_predictions_under_threshold_count, neg_predictions_under_threshold_count / neg_predictions_count * 100))

print("total negative prediction count: %s" % neg_predictions_count)
