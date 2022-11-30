# Copyright 2015 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
import json
import os
from io import StringIO, BytesIO

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from dataset import get_tg_dataset
from flask import Flask, render_template, request, send_file
from flask_cors import CORS
from model import *
from networkx.readwrite import json_graph
from utils import preselect_anchor


class Object(object):
    pass


# initialising Flask app
app = Flask(__name__, static_folder="./frontend/build/static", template_folder="./frontend/build")
CORS(app)

config = json.load(open("config.json"))

default_args = config["args"]

# creating indirect args object
args = Object()
for key, value in default_args.items():
    setattr(args, key, value)

# global variables
sigmoid = None
euclidean = None
datasets = {}
features_sets = {}
models = {}
models_sets = {}
device = "cpu"


def load_datasets(conf):
    global datasets
    global features_sets
    global device

    for dataset_name, dataset_config in conf["datasets"].items():
        datasets[dataset_name] = {}
        features_sets[dataset_name] = {}

        for task_name, task_config in conf["tasks"].items():
            if task_name not in dataset_config["supported_tasks"]:
                continue

            # loading dataset for each different amount of features
            selected_features_set = {}

            for category in dataset_config["feature_categories"]:
                features = category["features"]
                feature_set_name = category["name"]
                skip_loading = category["skip_loading"]

                if skip_loading is True:
                    continue

                selected_features_set[feature_set_name] = features

            data_list = get_tg_dataset(args, dataset_config["dataset_name"], dataset_config["duplicate_mapping_file"],
                                       task_name, selected_features_set=selected_features_set)

            # data
            print("Preparing dataset for direct attribute comparison...")
            for i, data in enumerate(data_list):
                preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')
                prepared_features_set = {}

                # baseline data preparation
                for category in dataset_config["feature_categories"]:
                    feature_set_name = category["name"]
                    skip_loading = category["skip_loading"]

                    if skip_loading is True:
                        continue

                    prepared_features = []
                    index_shift = 0
                    for j in range(len(data.feature_shape_set[feature_set_name])):
                        start_index = index_shift
                        end_index = start_index + data.feature_shape_set[feature_set_name][j]
                        index_shift = end_index
                        prepared_features.append(torch.index_select(data.features[feature_set_name], 1, torch.from_numpy(
                            np.asarray(range(data.features[feature_set_name].shape[1]))[start_index:end_index]).long()))

                    prepared_features_set[feature_set_name] = prepared_features

                data.prepared_features = prepared_features_set
                data = data.to(device)
                data_list[i] = data

            datasets[dataset_name][task_name] = data_list
            features_sets[dataset_name][task_name] = selected_features_set


def load_graph_models(conf):
    global datasets
    global models
    global sigmoid
    global euclidean
    global models_sets
    global device

    sigmoid = nn.Sigmoid()
    euclidean = nn.PairwiseDistance()

    model_id = 0

    for dataset_name, dataset_config in conf["datasets"].items():
        models[dataset_name] = {}
        models_sets[dataset_name] = {}

        for category in dataset_config["feature_categories"]:
            feature_set_name = category["name"]
            available_models = category["models"]
            skip_loading = category["skip_loading"]

            if skip_loading is True:
                continue

            if models[dataset_name].get(feature_set_name) is None:
                models[dataset_name][feature_set_name] = {}
                models_sets[dataset_name][feature_set_name] = {}

            for available_model in available_models:
                model_type = available_model["type"]
                model_task = available_model["task"]
                model_name = available_model["model_name"]
                display_name = available_model["display_name"]

                if models[dataset_name][feature_set_name].get(model_task) is None:
                    models[dataset_name][feature_set_name][model_task] = {}
                    models_sets[dataset_name][feature_set_name][model_task] = []

                # initializing model
                if model_type == "Baseline":
                    feature_shapes = datasets[dataset_name][model_task][0].feature_shape_set[feature_set_name]
                    output_dim = available_model["output_dim"]
                    hidden_dim = available_model["hidden_dim"]

                    model_instance = {
                        "main": globals()[model_type](feature_shapes=feature_shapes, hidden_dim=hidden_dim,
                                                      output_dim=output_dim, device=device).to(device),
                        "linear": globals()["Linear"](input_dim=len(feature_shapes), output_dim=1).to(device)
                    }

                    model_path = "./data/models/" + model_name
                    checkpoint = torch.load(model_path, map_location='cpu')
                    model_instance["main"].load_state_dict(checkpoint['model_state_dict'])

                    model_path = "./data/models/" + model_name + "-linear"
                    checkpoint = torch.load(model_path, map_location='cpu')
                    model_instance["linear"].load_state_dict(checkpoint['model_state_dict'])

                    # set both models to evaluation mode
                    model_instance["main"].eval()
                    model_instance["linear"].eval()
                elif model_type != "JC":
                    input_dim = datasets[dataset_name][model_task][0].features[feature_set_name].shape[1]

                    feature_dim = available_model["feature_dim"]
                    hidden_dim = available_model["hidden_dim"]
                    output_dim = available_model["output_dim"]
                    dropout = available_model["dropout"]
                    feature_pre = available_model["feature_pre"]

                    model_instance = globals()[model_type](input_dim=input_dim, feature_dim=feature_dim,
                                                           hidden_dim=hidden_dim, output_dim=output_dim,
                                                           feature_pre=feature_pre, layer_num=args.layer_num,
                                                           dropout=dropout).to(device)

                    # load the checkpoint
                    model_path = "./data/models/" + model_name
                    checkpoint = torch.load(model_path, map_location='cpu')

                    model_instance.load_state_dict(checkpoint['model_state_dict'])

                    # set the model to evaluation mode
                    model_instance.eval()

                unique_name = model_type + "_" + feature_set_name + "_" + model_task + "-" + str(model_id)

                models[dataset_name][feature_set_name][model_task][unique_name] = model_instance
                models_sets[dataset_name][feature_set_name][model_task].append({
                    "display_name": display_name,
                    "unique_name": unique_name
                })

                model_id += 1


@app.route("/available")
def get_available_items():
    global features_sets
    global models_sets
    global datasets

    response = {}

    for dataset_name, dataset in datasets.items():
        first_key = list(dataset.keys())[0]

        response[dataset_name] = {
            "subgraphs": [*range(len(dataset[first_key]))],
            "attributes": features_sets[dataset_name][first_key],
            "models": models_sets[dataset_name],
            "supported_tasks": config["datasets"][dataset_name]["supported_tasks"]
        }

    return {
        "datasets": response,
        "tasks": config["tasks"]
    }


@app.route("/subgraphs/<int:graph_num>")
def get_sub_graph_info(graph_num):
    global datasets

    dataset = request.args.get('dataset')
    task = request.args.get('task')

    subgraph_data = datasets[dataset][task][graph_num]

    duplicate_mapping = []
    if task == "NORP":
        duplicate_mapping = subgraph_data.graph_duplicate_node_mappings
    elif task == "LINK":
        duplicate_mapping = subgraph_data.removed_edges

    return {
        "graph": json_graph.node_link_data(subgraph_data.graph),
        "mapping": subgraph_data.mapping,
        "duplicate_mapping": duplicate_mapping,
        "grouped_person_attributes": subgraph_data.grouped_person_attributes
    }


@app.route("/subgraphs/<int:graph_num>/similarity/<int:node1>/<int:node2>")
def get_node_similarity(graph_num, node1, node2):
    global datasets
    global models
    global device
    global sigmoid
    global euclidean

    dataset = request.args.get('dataset')
    task = request.args.get('task')
    selected_model = request.args.get('model')
    selected_features = request.args.get('features')
    show_for_all = request.args.get('all')

    data = datasets[dataset][task][graph_num]

    if show_for_all == "true":
        node1 = [node1] * data.num_nodes
        node2 = [i for i in range(data.num_nodes)]
    else:
        node1 = [node1]
        node2 = [node2]

    if "Baseline_" in selected_model:
        model = models[dataset][selected_features][task][selected_model]["main"]
        linear = models[dataset][selected_features][task][selected_model]["linear"]

        feats_first = []
        feats_second = []
        for j in range(len(data.feature_shape_set[selected_features])):
            feats_first.append(torch.index_select(data.prepared_features[selected_features][j], 0,
                                                  torch.from_numpy(np.asarray(node1)).long()))
            feats_second.append(torch.index_select(data.prepared_features[selected_features][j], 0,
                                                   torch.from_numpy(np.asarray(node2)).long()))

        nodes_first_out = model(feats_first)
        nodes_second_out = model(feats_second)

        predictions = []
        for j in range(len(data.feature_shape_set[selected_features])):
            pred = euclidean(nodes_first_out[j], nodes_second_out[j])
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=-1)
        predictions = linear(predictions)
        predictions = sigmoid(predictions).flatten()
        prediction = predictions.tolist()
    elif "JC_" in selected_model:
        jc = nx.jaccard_coefficient(data.graph, [(node1[i], node2[i]) for i in range(len(node1))])
        prediction = []
        for u, v, c in jc:
            prediction.append(float(c))
    else:
        model = models[dataset][selected_features][task][selected_model]

        out = model(data, selected_features)

        nodes_first = torch.index_select(out, 0, torch.from_numpy(np.asarray(node1)).long().to(device))
        nodes_second = torch.index_select(out, 0, torch.from_numpy(np.asarray(node2)).long().to(device))

        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        pred = sigmoid(pred)

        prediction = pred.tolist()

    return {
        "predictions": prediction,
        "task": task,
        "request": {
            "nodes": [
                node1,
                node2
            ],
            "model": selected_model,
            "features": selected_features
        }
    }


@app.route("/subgraphs/<int:graph_num>/feature_similarity/<int:node1>/<int:node2>")
def get_feature_similarity(graph_num, node1, node2):
    global datasets
    global models
    global device
    global sigmoid
    global euclidean

    dataset = request.args.get('dataset')
    selected_model = request.args.get('model')
    selected_features = request.args.get('features')

    data = datasets[dataset][graph_num]
    model = models[selected_features][selected_model]["main"]

    feats_first = []
    feats_second = []
    for j in range(len(data.feature_shape_set[selected_features])):
        feats_first.append(torch.index_select(data.prepared_features[selected_features][j], 0,
                                              torch.from_numpy(np.asarray([node1])).long()))
        feats_second.append(torch.index_select(data.prepared_features[selected_features][j], 0,
                                               torch.from_numpy(np.asarray([node2])).long()))

    nodes_first_out = model(feats_first)
    nodes_second_out = model(feats_second)

    predictions = []
    for j in range(len(data.feature_shape_set[selected_features])):
        pred = euclidean(nodes_first_out[j], nodes_second_out[j])
        predictions.append(float(pred.data[0]))

    return {
        "predictions": predictions
    }


@app.route("/generate/svg")
def generate_svg():
    colors = request.args.getlist('colors')
    is_checked = request.args.get('checked')

    size = 100
    steps = 100 / len(colors)

    svg = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" ' \
          'version="1.1" height="20" width="20" viewBox="0 0 20 20">'
    svg += '<circle r="10" cx="10" cy="10" fill="' + colors[0] + '"/>'

    for i in range(len(colors) - 1):
        size -= steps
        svg += '<circle r="5" cx="10" cy="10" fill="transparent" stroke="' + colors[i + 1] + \
               '" stroke-width="10" stroke-dasharray="calc(' + str(size) + \
               ' * 31.4 / 100) 31.4" transform="rotate(-90) translate(-20)"/>'

    if is_checked == "true":
        svg += '<circle style="opacity: 0.15" r="10" cx="10" cy="10" fill="black"/>'
        svg += '<line x1="5" y1="11" x2="9.1" y2="14" style="stroke:white; stroke-width:1.8"/>'
        svg += '<line x1="8" y1="14" x2="15" y2="6" style="stroke:white; stroke-width:1.8"/>'

    svg += '</svg>'

    svg_io = StringIO()
    svg_io.write(svg)
    svg_io.seek(0)

    mem = BytesIO()
    mem.write(svg_io.getvalue().encode('utf-8'))
    mem.seek(0)

    return send_file(mem, mimetype='image/svg+xml')


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    return render_template('index.html')


port = os.getenv('PORT', '8004')
load_datasets(config)
load_graph_models(config)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(port), threaded=True)
