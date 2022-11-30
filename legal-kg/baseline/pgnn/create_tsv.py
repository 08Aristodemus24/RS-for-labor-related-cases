import os.path
import time
from random import shuffle

import numpy as np
import torch
from args import make_args
from dataset import get_tg_dataset
from model import *
from utils import preselect_anchor, get_duplicate_mask

# COMMAND: python create_tsv.py --model PGNN --layer_num 2 --epoch_num 4001 --dataset clean_more_than_subject_person_subjectdup --dataset_duplicates clean_more_than_subject_person_subjectdup.map

# parse args and print them
args = make_args()
print(args)

device = 'cpu'
model_name = "clean_more_than_subject_person_subjectdup_PGNN_4001_similarity_2layer1571089058.3082974_task62.pt"
dataset_name = args.dataset

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

num_features = data_list[0].x.shape[1]

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

# load the checkpoint
model_path = "./models/" + model_name
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# set model to training mode
model.eval()

tsv_items = []
tsv_names = []

for id, data in enumerate(data_list):
    # skip subgraphs without positive samples
    if data.mask_duplicate_positive_train.shape[1] == 0:
        continue

    if id != 39:
        continue

    out = model(data)

    for i in range(out.shape[0]):
        # if out.shape[1] != 16:
        #     continue

        line = ""
        for j in range(out.shape[1]):
            line += str(out[i][j].item()) + "\t"
        tsv_items.append(line[:-2])

        for name, node_id in data.mapping.items():
            if node_id == i:
                tsv_names.append(name)

with open("./data.tsv", "w") as file:
    for i in range(len(tsv_items)):
        file.write(tsv_items[i] + '\n')

with open("./metadata.tsv", "w") as file:
    for i in range(len(tsv_items)):
        file.write(tsv_names[i] + '\n')
