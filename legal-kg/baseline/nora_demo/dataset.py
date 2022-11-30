import torch
import networkx as nx
import numpy as np
import torch.utils.data
import json
import pickle
import os.path

from networkx.readwrite import json_graph
from torch_geometric.data import Data

from utils import precompute_dist_data, get_duplicate_mask, convert_mask_to_node_names, \
    convert_mask_from_node_names, print_progress_bar, duplicate_edges


def get_tg_dataset(args, dataset_name, dataset_duplicates, task, selected_features_set={}):
    """
    Loading the dataset and prepare it for later use.
    :param task_name:
    :param dataset_duplicates:
    :param selected_features_set:
    :param args: parsed arguments
    :param dataset_name: name of dataset in folder "../dataprep/data/"
    :return: data list array for each subgraph
    """
    try:
        dataset = load_tg_dataset(dataset_name, dataset_duplicates, selected_features_set=selected_features_set)
    except:
        raise NotImplementedError

    # loading dataset from folder
    f1_name = './data/datasets_cache/' + dataset_name + '_train-' + task + '.dat'
    f2_name = './data/datasets_cache/' + dataset_name + '_val-' + task + '.dat'
    f3_name = './data/datasets_cache/' + dataset_name + '_test-' + task + '.dat'

    # load lists from cache
    with open(f1_name, 'rb') as f1, open(f2_name, 'rb') as f2, open(f3_name, 'rb') as f3:
        duplicate_train_list = pickle.load(f1)
        duplicate_val_list = pickle.load(f2)
        duplicate_test_list = pickle.load(f3)
        print('Cache loaded!')

    data_list = []
    for i, data in enumerate(dataset):
        # convert node names to temporary node ids in subgraph (because node ids seem to be random every time)
        data.mask_duplicate_positive_train = convert_mask_from_node_names(duplicate_train_list[i], data.mapping)
        data.mask_duplicate_positive_val = convert_mask_from_node_names(duplicate_val_list[i], data.mapping)
        data.mask_duplicate_positive_test = convert_mask_from_node_names(duplicate_test_list[i], data.mapping)

        if task == "LINK":
            data.edge_index = torch.from_numpy(duplicate_edges(
                np.concatenate((data.mask_duplicate_positive_train, data.mask_duplicate_positive_val), axis=1))).long()

            removed_edges = np.concatenate((data.mask_duplicate_positive_train, data.mask_duplicate_positive_val,
                                            data.mask_duplicate_positive_test), axis=1)
            removed_edges_mapping = []
            for j in range(removed_edges.shape[1]):
                removed_edges_mapping.append([int(removed_edges[0][j]), int(removed_edges[1][j])])
            data.removed_edges = removed_edges_mapping

            dists = precompute_dist_data(data.mask_duplicate_positive_train, data.num_nodes,
                                         approximate=args.approximate)
        else:
            dists = precompute_dist_data(data.edge_index.numpy(), data.num_nodes, approximate=args.approximate)

        print_progress_bar(i + 1, len(dataset), prefix="subgraphs: ", length=50)

        # Creates a Tensor from a numpy.ndarray.
        data.dists = torch.from_numpy(dists).float()

        data.duplicate_index = torch.from_numpy(
            np.concatenate((data.mask_duplicate_positive_train,
                            data.mask_duplicate_positive_test,
                            data.mask_duplicate_positive_val),
                           axis=1)).long()

        data_list.append(data)

    return data_list


def nx_to_tg_data(graphs, feature_set, duplicate_node_mappings, grouped_tuples, feature_shape_set, edge_labels=None):
    """
    Function for converting raw data to Data object.
    :param feature_shape_set:
    :param feature_set:
    :param graphs: array including the networkx graph object for each subgraph
    :param duplicate_node_mappings: array including the mapping between duplicate nodes in each subgraph
    :param edge_labels: array including the edge labels between nodes in each subgraph
    :return: data_list array for each subgraph with prepared data
    """
    data_list = []

    print("processing ", len(graphs), " graphs")
    for i in range(len(graphs)):
        graph = graphs[i].copy()
        graph.remove_edges_from(graph.selfloop_edges())

        # create mapping for relabeling graph nodes
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))

        # relabel and sum up mapping between duplicate nodes
        graph_duplicate_node_mappings = []
        grouped_person_attributes = {}

        for key, value in mapping.items():
            if key in duplicate_node_mappings.keys():
                if duplicate_node_mappings[key] in mapping.keys():
                    graph_duplicate_node_mappings.append((value, mapping[duplicate_node_mappings[key]]))

            grouped_person_attributes[value] = grouped_tuples.get(key)

        nx.relabel_nodes(graph, mapping, copy=False)

        feature = {}
        for feature_set_name, features in feature_set.items():
            x = np.zeros(features[i].shape)
            graph_nodes = list(graph.nodes)

            # get feature vector x for each node in subgraph
            for m in range(features[i].shape[0]):
                x[graph_nodes[m]] = features[i][m]
            x = torch.from_numpy(x).float()

            feature[feature_set_name] = x

        # get edges => duplicating each existent edge from one side to other side (e.g. 8 => 1 && 8 <= 1)
        edge_index = np.array(list(graph.edges))
        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)

        # get duplicate index
        duplicate_index = None
        try:
            duplicate_index = np.array(list(graph_duplicate_node_mappings))
            # duplicate_index = np.concatenate((duplicate_index, duplicate_index[:, ::-1]), axis=0)
            duplicate_index = torch.from_numpy(duplicate_index).long().permute(1, 0)
        except:
            duplicate_index = torch.from_numpy(duplicate_index).long()

        # create Data object from torch_geometric, including all data for subgraph
        data = Data(x=feature[list(feature_set.keys())[0]], edge_index=edge_index)
        data.graph = graph

        # get edge_labels
        if edge_labels[0] is not None:
            edge_label = edge_labels[i]
            mask_link_positive = np.stack(np.nonzero(edge_label))
            data.mask_link_positive = mask_link_positive
            mask_duplicate_positive = np.stack(np.nonzero(edge_label))
            data.mask_duplicate_positive = mask_duplicate_positive
            data.mapping = mapping
            data.features = feature
            data.grouped_person_attributes = grouped_person_attributes
            data.graph_duplicate_node_mappings = graph_duplicate_node_mappings
            data.duplicate_index = duplicate_index
            data.feature_shape_set = feature_shape_set
        data_list.append(data)

    return data_list


def load_graphs_v2(dataset_str, dataset_duplicates_str, selected_features_set={}):
    """
    Main data load function from relevant files by using more advanced features. (v2)
    :param selected_features_set: dict of list of attributes, which should be used for feature vector generation
    :param dataset_str: name of dataset in folder "../dataprep/data/"
    :param dataset_duplicates_str: file name of related ".map" file in "../dataprep/generated/"
    :return: separated arrays including different data for each subgraph
    """

    dataset_dir = '../dataprep/data/' + dataset_str
    print("Loading data...")

    G = json_graph.node_link_graph(json.load(open(dataset_dir + "-G.json")))

    # loading duplicate mappings
    with open("../dataprep/generated/" + dataset_duplicates_str, 'rb') as file:
        duplicate_node_mappings = pickle.load(file)

    train_ids = [n for n in G.nodes()]
    edge_labels = np.array([[0, 1] for i in train_ids])

    print("Using only features..")
    with open(dataset_dir + "-feats.dat", 'rb') as file:
        attributes_raw_features = pickle.load(file)

    grouped_tuples = json.load(open(dataset_dir + "-grouped.json"))

    node_dict = {}
    for id, node in enumerate(G.nodes()):
        node_dict[node] = id

    print("Finding connected components..")
    comps = [comp for comp in nx.connected_components(G) if len(comp) > 10]
    print("Finding subgraphs..")
    graphs = [G.subgraph(comp) for comp in comps]
    print("Number of Graphs in input:")
    print(len(graphs))

    id_all = []
    for comp in comps:
        id_temp = []
        for node in comp:
            id = node_dict[node]
            id_temp.append(id)
        id_all.append(np.array(id_temp))

    feat_id_map = json.load(open(dataset_dir + "-id_map.json"))
    feat_id_map = {id: val for id, val in feat_id_map.items()}

    feature_shape_set = {}
    feature_set = {}

    for feature_set_name, selected_features in selected_features_set.items():
        feats = None
        feature_shapes = []

        if len(selected_features) == 0:
            feature_shapes.append(1)
            feats = np.ones((attributes_raw_features[list(attributes_raw_features.keys())[0]].shape[0], 1))

        for attribute_name, attribute_feature_array in attributes_raw_features.items():
            if attribute_name in selected_features:
                if feats is None:
                    feats = attribute_feature_array
                else:
                    feats = np.hstack((feats, attribute_feature_array))

                feature_shapes.append(attribute_feature_array.shape[1])

        train_feats = feats[[feat_id_map[id] for id in train_ids]]

        print("Creating features for: " + feature_set_name)
        features = [train_feats[id_temp, :] for id_temp in id_all]

        feature_shape_set[feature_set_name] = feature_shapes
        feature_set[feature_set_name] = features

    return graphs, feature_set, edge_labels, dict(
        duplicate_node_mappings), grouped_tuples, feature_shape_set


def load_tg_dataset(name='udbms', dataset_duplicates_str="person_duplicate", selected_features_set={}):
    """
    Loading the graph from dataset and convert it to expected format.
    :param selected_features_set:
    :param name: name of dataset in folder "../dataprep/data/"
    :param dataset_duplicates_str: file name of related ".map" file in "../dataprep/generated/"
    :return: data list for each subgraph calculated by nx_to_tg_data method
    """
    print("Load tg dataset")

    graphs, feature_set, edge_labels, duplicate_node_mappings, \
    grouped_tuples, feature_shape_set = load_graphs_v2(name, dataset_duplicates_str, selected_features_set)

    return nx_to_tg_data(graphs, feature_set, duplicate_node_mappings,
                         grouped_tuples, feature_shape_set, edge_labels)
