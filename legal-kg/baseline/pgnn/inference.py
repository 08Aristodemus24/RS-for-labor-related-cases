import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import pickle
import os
import torch.nn as nn
from args import make_args
from dataset import get_tg_dataset
from model import *
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from utils import preselect_anchor, get_duplicate_mask, precision_recall_curve

np.random.seed(123)
np.random.seed()

# COMMAND: python inference.py --model PGNN_plus --layer_num 2 --dataset clean_more_than_subject_person_subjectdup --dataset_duplicates clean_more_than_subject_person_subjectdup.map
# COMMAND: python inference.py --model PGNN --layer_num 2 --dataset clean_more_than_subject_person_subjectdup --dataset_duplicates clean_more_than_subject_person_subjectdup.map
# COMMAND: python inference.py --model SAGE --layer_num 2 --dataset clean_more_than_subject_person_subjectdup --dataset_duplicates clean_more_than_subject_person_subjectdup.map
# COMMAND: python inference.py --model GIN --layer_num 2 --dataset clean_more_than_subject_person_subjectdup --dataset_duplicates clean_more_than_subject_person_subjectdup.map
# COMMAND: python inference.py --model GCN --layer_num 2 --dataset clean_more_than_subject_person_subjectdup --dataset_duplicates clean_more_than_subject_person_subjectdup.map

# additional arguments
comment = ""
device = 'cpu'
model_name = "clean_more_than_subject_person_subjectdup_PGNN_4001_similarity_2layer1571852677.5620842_task68.pt"
iteration_count = 100
inference = False

# draw roc curve when AUC is in range of given value
draw_roc_curve = False
drc_value = 0.5988
drc_value_threshold = 0.005

args = make_args()

result_test = []
time1 = time.time()
data_list = get_tg_dataset(args, args.dataset, use_cache=args.cache, remove_feature=args.rm_feature, inference=inference,
                           selected_features=[]
                           )

time2 = time.time()
print(args.dataset, 'load time', time2 - time1)

num_features = data_list[0].x.shape[1]
print("Number of features: " + str(num_features))
num_node_classes = None
num_graph_classes = None
if 'y' in data_list[0].__dict__ and data_list[0].y is not None:
    num_node_classes = max([data.y.max().item() for data in data_list]) + 1
if 'y_graph' in data_list[0].__dict__ and data_list[0].y_graph is not None:
    num_graph_classes = max([data.y_graph.numpy()[0] for data in data_list]) + 1
print('Dataset', args.dataset, 'Graph', len(data_list), 'Feature', num_features, 'Node Class', num_node_classes,
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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
loss_func = nn.BCEWithLogitsLoss()
out_act = nn.Sigmoid()

# load the checkpoint
model_path = "./models/" + model_name
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# set the model on evaluate mode
model.eval()

results_auc_score = np.asarray([])
results_acc_score = np.asarray([])
results_accuracy_score_pos = np.asarray([])
results_accuracy_score_neg = np.asarray([])
results_average_precision = np.asarray([])

results_jc_auc_score = np.asarray([])
results_jc_acc_score = np.asarray([])
results_jc_accuracy_score_pos = np.asarray([])
results_jc_accuracy_score_neg = np.asarray([])
results_jc_average_precision = np.asarray([])

prf1_results = {}

for iteration_id in range(iteration_count):
    total_labels = np.asarray([])
    total_labels_pos = np.asarray([])
    total_labels_neg = np.asarray([])

    total_predictions = np.asarray([])
    total_predictions_pos = np.asarray([])
    total_predictions_neg = np.asarray([])

    total_predictions_jc = np.asarray([])
    total_predictions_jc_pos = np.asarray([])
    total_predictions_jc_neg = np.asarray([])

    for id, data in enumerate(data_list):
        if inference is False:
            get_duplicate_mask(data, resplit=False, use_jaccard_ranking=False)
            preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')

        out = model(data)

        # test
        duplicates_mask_test = np.concatenate((data.mask_duplicate_positive_test, data.mask_duplicate_negative_test),
                                              axis=-1)
        nodes_first = torch.index_select(out, 0, torch.from_numpy(duplicates_mask_test[0, :]).long().to(device))
        nodes_second = torch.index_select(out, 0, torch.from_numpy(duplicates_mask_test[1, :]).long().to(device))

        pred = torch.sum(nodes_first * nodes_second, dim=-1)

        label_positive = torch.ones([data.mask_duplicate_positive_test.shape[1], ], dtype=pred.dtype)
        label_negative = torch.zeros([data.mask_duplicate_negative_test.shape[1], ], dtype=pred.dtype)
        label = torch.cat((label_positive, label_negative)).to(device)

        try:
            # appending all / positive / negative labels and predictions to np arrays
            positive_sample_labels = label.flatten().cpu().numpy()
            positive_sample_labels = positive_sample_labels[:data.mask_duplicate_positive_test.shape[1]]
            total_labels_pos = np.append(total_labels_pos, positive_sample_labels)

            negative_sample_labels = label.flatten().cpu().numpy()
            negative_sample_labels = negative_sample_labels[data.mask_duplicate_positive_test.shape[1]:]
            total_labels_neg = np.append(total_labels_neg, negative_sample_labels)

            total_labels = np.append(total_labels, label.flatten().cpu().numpy())
            total_predictions = np.append(total_predictions, pred.flatten().data.cpu().numpy())

            positive_predictions = out_act(pred).flatten().data.cpu().numpy()[
                                   :data.mask_duplicate_positive_test.shape[1]]
            total_predictions_pos = np.append(total_predictions_pos, positive_predictions)

            negative_predictions = out_act(pred).flatten().data.cpu().numpy()[
                                   data.mask_duplicate_positive_test.shape[1]:]
            total_predictions_neg = np.append(total_predictions_neg, negative_predictions)

            # calculating jc score for each testing pair and append to np array
            try:
                g_pairs = []
                g_pairs_pos = []
                g_pairs_neg = []

                # getting positive pairs
                for h in range(data.mask_duplicate_positive_test.shape[1]):
                    first = data.mask_duplicate_positive_test[0][h]
                    second = data.mask_duplicate_positive_test[1][h]
                    g_pairs_pos.append((first, second))
                    g_pairs.append((first, second))

                # getting negative pairs
                for h in range(data.mask_duplicate_negative_test.shape[1]):
                    first = data.mask_duplicate_negative_test[0][h]
                    second = data.mask_duplicate_negative_test[1][h]
                    g_pairs_neg.append((first, second))
                    g_pairs.append((first, second))

                # calculating jc score for all / positive / negative pairs
                jc = nx.jaccard_coefficient(data.graph, g_pairs)
                jc_pos = nx.jaccard_coefficient(data.graph, g_pairs_pos)
                jc_neg = nx.jaccard_coefficient(data.graph, g_pairs_neg)

                for u, v, c in jc:
                    total_predictions_jc = np.append(total_predictions_jc, c)

                for u, v, c in jc_pos:
                    total_predictions_jc_pos = np.append(total_predictions_jc_pos, c)

                for u, v, c in jc_neg:
                    total_predictions_jc_neg = np.append(total_predictions_jc_neg, c)

            except Exception as e:
                print(e)

        except Exception as e:
            print(e)

    # calculating auc score
    auc_score = roc_auc_score(total_labels, total_predictions)
    jc_auc_score = roc_auc_score(total_labels, total_predictions_jc)

    # calculating average precision
    average_precision = average_precision_score(total_labels, total_predictions)
    jc_average_precision = average_precision_score(total_labels, total_predictions_jc)

    # calculating accuracy score
    acc_score = accuracy_score(total_labels, np.rint(out_act(torch.from_numpy(total_predictions)).numpy()))
    jc_acc_score = accuracy_score(total_labels, np.rint(total_predictions_jc))

    # positive accuracy score over threshold 0.5 (np.rint)
    accuracy_score_pos = accuracy_score(total_labels_pos, np.rint(total_predictions_pos))
    jc_accuracy_score_pos = accuracy_score(total_labels_pos, np.rint(total_predictions_jc_pos))

    # negative accuracy score over threshold 0.5 (np.rint)
    accuracy_score_neg = accuracy_score(total_labels_neg, np.rint(total_predictions_neg))
    jc_accuracy_score_neg = accuracy_score(total_labels_neg, np.rint(total_predictions_jc_neg))

    # calculating precision/recall
    precision, recall, thresholds = precision_recall_curve(total_labels,
                                                           out_act(torch.from_numpy(total_predictions)).numpy())

    # calculating and appending f1/precision/recall scores
    highest_f1 = 0
    f1_threshold = 0
    for k in range(len(precision)):
        precision_value = precision[k]
        recall_value = recall[k]
        threshold_value = thresholds[k]

        f1_score = 2 * (precision_value * recall_value) / (precision_value + recall_value)
        if np.isnan(f1_score):
            f1_score = 0

        if f1_score > highest_f1:
            highest_f1 = f1_score
            f1_threshold = threshold_value

        if prf1_results.get(threshold_value) is None:
            prf1_results[threshold_value] = {
                "f1": np.asarray([]),
                "threshold": np.asarray([]),
                "precision": np.asarray([]),
                "recall": np.asarray([])
            }

        prf1_results[threshold_value] = {
            "f1": np.append(prf1_results[threshold_value]["f1"], f1_score),
            "precision": np.append(prf1_results[threshold_value]["precision"], precision_value),
            "recall": np.append(prf1_results[threshold_value]["recall"], recall_value),
        }

    print()
    print("=========================", "Iteration", iteration_id, "===")
    print()
    print("AUC:")
    print(auc_score)
    print("Accuracy Score")
    print(acc_score)
    print("Positive Samples Accuracy:")
    print(accuracy_score_pos)
    print("Negative Samples Accuracy:")
    print(accuracy_score_neg)
    print("Average Precision:")
    print(average_precision)
    print("max F1/Threshold:")
    print(highest_f1, "(" + str(f1_threshold) + ")")

    print()
    print("JC AUC:")
    print(jc_auc_score)
    print("Accuracy Score")
    print(jc_acc_score)
    print("Positive Samples Accuracy:")
    print(jc_accuracy_score_pos)
    print("Negative Samples Accuracy:")
    print(jc_accuracy_score_neg)
    print("Average Precision:")
    print(jc_average_precision)

    results_auc_score = np.append(results_auc_score, auc_score)
    results_acc_score = np.append(results_acc_score, acc_score)
    results_accuracy_score_pos = np.append(results_accuracy_score_pos, accuracy_score_pos)
    results_accuracy_score_neg = np.append(results_accuracy_score_neg, accuracy_score_neg)
    results_average_precision = np.append(results_average_precision, average_precision)

    results_jc_auc_score = np.append(results_jc_auc_score, jc_auc_score)
    results_jc_acc_score = np.append(results_jc_acc_score, jc_acc_score)
    results_jc_accuracy_score_pos = np.append(results_jc_accuracy_score_pos, jc_accuracy_score_pos)
    results_jc_accuracy_score_neg = np.append(results_jc_accuracy_score_neg, jc_accuracy_score_neg)
    results_jc_average_precision = np.append(results_jc_average_precision, jc_average_precision)

    if draw_roc_curve is True and drc_value - drc_value_threshold < auc_score < drc_value + drc_value_threshold:
        directory = "./inference_results/roc/" + model_name.replace(".", "_")
        if not os.path.exists(directory):
            os.makedirs(directory)

        fpr, tpr, threshold = metrics.roc_curve(total_labels, total_predictions)
        roc_auc = metrics.auc(fpr, tpr)

        with open(directory + "/roc_curve-" + str(iteration_id) + "_auc" + str(roc_auc) + ".dat", "wb") as file:
            pickle.dump({
                "fpr": fpr,
                "tpr": tpr,
                "threshold": threshold,
                "roc_auc": roc_auc
            }, file, protocol=4)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(directory + "/roc_curve-" + str(iteration_id) + "_auc" + str(roc_auc) + ".pdf", bbox_inches="tight")
        plt.clf()

BOLD = '\033[1m'
END = '\033[0m'

# calculate overall maximum f1 average score
threshold_f1 = 0
max_avg_f1 = 0
std_f1 = 0
avg_precision = 0
std_precision = 0
avg_recall = 0
std_recall = 0

for threshold, items in prf1_results.items():
    average_f1 = np.average(items["f1"])

    if average_f1 > max_avg_f1:
        threshold_f1 = threshold
        max_avg_f1 = average_f1
        std_f1 = np.std(items["f1"])
        avg_precision = np.average(items["precision"])
        std_precision = np.std(items["precision"])
        avg_recall = np.average(items["recall"])
        std_recall = np.std(items["recall"])

print()
print()
print("=========================", "AVG RESULT", "(" + str(iteration_count) + " iterations)", "===")
print()
print(BOLD + "AUC:" + END)
print(np.average(results_auc_score), "(std " + str(np.std(results_auc_score)) + ")")
print(BOLD + "Accuracy Score:" + END)
print(np.average(results_acc_score), "(std " + str(np.std(results_acc_score)) + ")")
print(BOLD + "Positive Samples Accuracy:" + END)
print(np.average(results_accuracy_score_pos), "(std " + str(np.std(results_accuracy_score_pos)) + ")")
print(BOLD + "Negative Samples Accuracy:" + END)
print(np.average(results_accuracy_score_neg), "(std " + str(np.std(results_accuracy_score_neg)) + ")")
print(BOLD + "Average Precision:" + END)
print(np.average(results_average_precision), "(std " + str(np.std(results_average_precision)) + ")")
print(BOLD + "Highest Average F1:" + END)
print("=> threshold:", threshold_f1)
print("=> f1:", max_avg_f1, "(std " + str(std_f1) + ")")
print("=> precision:", avg_precision, "(std " + str(std_precision) + ")")
print("=> recall:", avg_recall, "(std " + str(std_recall) + ")")

print()
print(BOLD + "JC AUC:" + END)
print(np.average(results_jc_auc_score), "(std " + str(np.std(results_jc_auc_score)) + ")")
print(BOLD + "Accuracy Score:" + END)
print(np.average(results_jc_acc_score), "(std " + str(np.std(results_jc_acc_score)) + ")")
print(BOLD + "Positive Samples Accuracy:" + END)
print(np.average(results_jc_accuracy_score_pos), "(std " + str(np.std(results_jc_accuracy_score_pos)) + ")")
print(BOLD + "Negative Samples Accuracy:" + END)
print(np.average(results_jc_accuracy_score_neg), "(std " + str(np.std(results_jc_accuracy_score_neg)) + ")")
print(BOLD + "Average Precision:" + END)
print(np.average(results_jc_average_precision), "(std " + str(np.std(results_jc_average_precision)) + ")")

result_path = "./inference_results/results/results_" + model_name + "_" + str(
    iteration_count) + "iterations" + comment + ".dat"

with open(result_path, "wb") as file:
    pickle.dump({
        "iteration_count": iteration_count,
        "results_auc_score": results_auc_score,
        "results_acc_score": results_acc_score,
        "results_accuracy_score_pos": results_accuracy_score_pos,
        "results_accuracy_score_neg": results_accuracy_score_neg,
        "results_average_precision": results_average_precision,
        "results_prf1": prf1_results,
        "results_jc_auc_score": results_jc_auc_score,
        "results_jc_acc_score": results_jc_acc_score,
        "results_jc_accuracy_score_pos": results_jc_accuracy_score_pos,
        "results_jc_accuracy_score_neg": results_jc_accuracy_score_neg,
        "results_jc_average_precision": results_jc_average_precision
    }, file, protocol=4)
