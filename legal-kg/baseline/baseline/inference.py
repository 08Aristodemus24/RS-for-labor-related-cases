import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import pickle
import os
from dataset import get_tg_dataset
from model import *
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from utils import get_duplicate_mask, precision_recall_curve

np.random.seed(123)
np.random.seed()

# additional arguments
comment = "roc_run"
dataset_name = "clean_more_than_subject_person_subjectdup"
dataset_duplicates = dataset_name + ".map"
selected_features = ["birthYear", "birthDate", "birthPlace", "deathYear", "deathDate", "activeYearsStartYear",
                     "deathPlace", "activeYearsEndYear", "almaMater", "deathCause", "restingPlace", "education",
                     "residence", "religion", "nationality", "stateOfOrigin", "knownFor", "party", "ethnicity", "award",
                     "networth", "hometown", "employer", "board", "citizenship"]

cuda = 0
batch_size = 8
epoch_log = 10
device = 'cpu'
model_name = "clean_more_than_subject_person_subjectdup_task64.pt"
iteration_count = 100
inference = False

# draw roc curve when AUC is in range of given value
draw_roc_curve = True
drc_value = 0.66582
drc_value_threshold = 0.005

data_list = get_tg_dataset(dataset_name, dataset_duplicates, inference=inference, selected_features=selected_features)

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

# load the checkpoint
model_path = "./models/" + model_name
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
linear_epoch = checkpoint['epoch']
linear_loss = checkpoint['loss']

model_path = "./models/" + model_name + "-linear"
checkpoint = torch.load(model_path, map_location='cpu')
linear.load_state_dict(checkpoint['model_state_dict'])
linear_epoch = checkpoint['epoch']
linear_loss = checkpoint['loss']

# set the model on evaluate mode
model.eval()
linear.eval()

# initialize sigmoid and cosine function
out_act = nn.Sigmoid()
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
euclidean = nn.PairwiseDistance()

results_auc_score = np.asarray([])
results_acc_score = np.asarray([])
results_accuracy_score_pos = np.asarray([])
results_accuracy_score_neg = np.asarray([])
results_average_precision = np.asarray([])

prf1_results = {}

for iteration_id in range(iteration_count):
    total_labels = np.asarray([])
    total_labels_pos = np.asarray([])
    total_labels_neg = np.asarray([])

    total_predictions = np.asarray([])
    total_predictions_pos = np.asarray([])
    total_predictions_neg = np.asarray([])

    for id, data in enumerate(data_list):
        # skip subgraphs without positive samples
        if data.mask_duplicate_positive_test.shape[1] == 0:
            continue

        if inference is False:
            get_duplicate_mask(data, resplit=False, use_jaccard_ranking=False)

        # test
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
            feats_first.append(
                torch.index_select(data.prepared_features[j], 0,
                                   torch.from_numpy(duplicate_mask_test[0, :]).long()))
            feats_second.append(
                torch.index_select(data.prepared_features[j], 0,
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

        # appending all / positive / negative labels to np arrays
        total_labels = np.append(total_labels, label.flatten().cpu().numpy())

        positive_sample_labels = label.flatten().cpu().numpy()
        positive_sample_labels = positive_sample_labels[:data.mask_duplicate_positive_test.shape[1]]
        total_labels_pos = np.append(total_labels_pos, positive_sample_labels)

        negative_sample_labels = label.flatten().cpu().numpy()
        negative_sample_labels = negative_sample_labels[data.mask_duplicate_positive_test.shape[1]:]
        total_labels_neg = np.append(total_labels_neg, negative_sample_labels)

        # appending all / positive / negative predictions to np arrays
        total_predictions = np.append(total_predictions, predictions.flatten().data.cpu().numpy())

        positive_predictions = out_act(predictions).flatten().data.cpu().numpy()[
                               :data.mask_duplicate_positive_test.shape[1]]
        total_predictions_pos = np.append(total_predictions_pos, positive_predictions)

        negative_predictions = out_act(predictions).flatten().data.cpu().numpy()[
                               data.mask_duplicate_positive_test.shape[1]:]
        total_predictions_neg = np.append(total_predictions_neg, negative_predictions)

    # calculating auc score
    auc_score = roc_auc_score(total_labels, total_predictions)

    # calculating average precision
    average_precision = average_precision_score(total_labels, total_predictions)

    # calculating accuracy score
    acc_score = accuracy_score(total_labels, np.rint(out_act(torch.from_numpy(total_predictions)).numpy()))

    # positive accuracy score over threshold 0.5 (np.rint)
    accuracy_score_pos = accuracy_score(total_labels_pos, np.rint(total_predictions_pos))

    # negative accuracy score over threshold 0.5 (np.rint)
    accuracy_score_neg = accuracy_score(total_labels_neg, np.rint(total_predictions_neg))

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

    results_auc_score = np.append(results_auc_score, auc_score)
    results_acc_score = np.append(results_acc_score, acc_score)
    results_accuracy_score_pos = np.append(results_accuracy_score_pos, accuracy_score_pos)
    results_accuracy_score_neg = np.append(results_accuracy_score_neg, accuracy_score_neg)
    results_average_precision = np.append(results_average_precision, average_precision)

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
        "results_prf1": prf1_results
    }, file, protocol=4)
