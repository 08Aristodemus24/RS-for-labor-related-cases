import json
import os.path
import pickle

import networkx as nx
import numpy as np
import torch
import torch.utils.data
from networkx.readwrite import json_graph
from torch_geometric.data import Data
from utils import precompute_dist_data, get_duplicate_mask, convert_mask_to_node_names, \
    convert_mask_from_node_names, print_progress_bar


def get_tg_dataset(args, dataset_name, use_cache=False, remove_feature=False, inference=False, selected_features=[]):
    """
    Loading the dataset and prepare it for later use.
    :param selected_features: labels of feature attributes to be included in the feature vectors
    :param inference: whether is in inference mode or nor (loading same cached negative samples for evaluation)
    :param args: parsed arguments
    :param dataset_name: name of dataset in folder "../dataprep/data/"
    :param use_cache: boolean whether caching is enabled or not
    :param remove_feature: boolean whether node features will be used or not
    :return: data list array for each subgraph
    """
    try:
        dataset = load_tg_dataset(dataset_name, args.dataset_duplicates, selected_features=selected_features)
    except:
        raise NotImplementedError

    # creating dataset caching directory if required
    if not os.path.isdir('datasets/cache'):
        os.mkdir('datasets/cache')
    f1_name = 'datasets/cache/' + dataset_name + str(args.approximate) + '_duplicates_train.dat'
    f2_name = 'datasets/cache/' + dataset_name + str(args.approximate) + '_duplicates_val.dat'
    f3_name = 'datasets/cache/' + dataset_name + str(args.approximate) + '_duplicates_test.dat'

    f4_name = 'datasets/cache/' + dataset_name + str(args.approximate) + '_duplicates_train_neg.dat'
    f5_name = 'datasets/cache/' + dataset_name + str(args.approximate) + '_duplicates_val_neg.dat'
    f6_name = 'datasets/cache/' + dataset_name + str(args.approximate) + '_duplicates_test_neg.dat'

    if use_cache is False or os.path.isfile(f1_name) is False:
        print("No cache..")
        data_list = []

        duplicate_train_list = []
        duplicate_val_list = []
        duplicate_test_list = []
        duplicate_train_neg_list = []
        duplicate_val_neg_list = []
        duplicate_test_neg_list = []

        print("")
        print("Preparing data (dists calculation)...")
        for i, data in enumerate(dataset):
            # create positive and negative duplication samples for each subgraph (train / test / val)
            get_duplicate_mask(data, resplit=True, use_jaccard_ranking=True)

            # save them to lists for caching, convert temporary node ids to unique names
            duplicate_train_list.append(convert_mask_to_node_names(data.mask_duplicate_positive_train, data.mapping))
            duplicate_val_list.append(convert_mask_to_node_names(data.mask_duplicate_positive_val, data.mapping))
            duplicate_test_list.append(convert_mask_to_node_names(data.mask_duplicate_positive_test, data.mapping))
            duplicate_train_neg_list.append(
                convert_mask_to_node_names(data.mask_duplicate_negative_train, data.mapping))
            duplicate_val_neg_list.append(convert_mask_to_node_names(data.mask_duplicate_negative_val, data.mapping))
            duplicate_test_neg_list.append(convert_mask_to_node_names(data.mask_duplicate_negative_test, data.mapping))

            # convert edge_index from previous calculation to torch tensor
            data.edge_index = torch.from_numpy(data.edge_index.numpy()).long()

            # pre-compute the distances between all nodes in the subgraph to each other
            dists = precompute_dist_data(data.edge_index.numpy(), data.num_nodes, approximate=args.approximate)
            print_progress_bar(i + 1, len(dataset), prefix="subgraphs: ", length=50)

            # Creates a Tensor from a numpy.ndarray.
            data.dists = torch.from_numpy(dists).float()

            # set the duplicate_index based on the positive sample masks
            data.duplicate_index = torch.from_numpy(
                np.concatenate((data.mask_duplicate_positive_train,
                                data.mask_duplicate_positive_test,
                                data.mask_duplicate_positive_val),
                               axis=1)).long()

            if remove_feature:
                data.x = torch.ones((data.x.shape[0], 1))

            data_list.append(data)

        # save training, validation and testing lists to cache
        with open(f1_name, 'wb') as f1, open(f2_name, 'wb') as f2, open(f3_name, 'wb') as f3, \
                open(f4_name, 'wb') as f4, open(f5_name, 'wb') as f5, open(f6_name, 'wb') as f6:
            pickle.dump(duplicate_train_list, f1, protocol=4)
            pickle.dump(duplicate_val_list, f2, protocol=4)
            pickle.dump(duplicate_test_list, f3, protocol=4)
            pickle.dump(duplicate_train_neg_list, f4, protocol=4)
            pickle.dump(duplicate_val_neg_list, f5, protocol=4)
            pickle.dump(duplicate_test_neg_list, f6, protocol=4)
            print('Cache saved!')
    else:
        # load lists from cache
        with open(f1_name, 'rb') as f1, open(f2_name, 'rb') as f2, open(f3_name, 'rb') as f3, \
                open(f4_name, 'rb') as f4, open(f5_name, 'rb') as f5, open(f6_name, 'rb') as f6:
            duplicate_train_list = pickle.load(f1)
            duplicate_val_list = pickle.load(f2)
            duplicate_test_list = pickle.load(f3)
            try:
                duplicate_train_neg_list = pickle.load(f4)
                duplicate_val_neg_list = pickle.load(f5)
                duplicate_test_neg_list = pickle.load(f6)
            except Exception as e:
                print("Couldn't find saved negative sample sets.")

            print('Cache loaded!')

        data_list = []

        print("")
        print("Preparing data (dists calculation)...")
        train_samples = 0
        test_samples = 0
        val_samples = 0
        total_nodes = 0
        total_edges = 0

        for i, data in enumerate(dataset):
            # convert node names to temporary node ids in subgraph (because node ids seem to be random every time)
            data.mask_duplicate_positive_train = convert_mask_from_node_names(duplicate_train_list[i], data.mapping)
            data.mask_duplicate_positive_val = convert_mask_from_node_names(duplicate_val_list[i], data.mapping)
            data.mask_duplicate_positive_test = convert_mask_from_node_names(duplicate_test_list[i], data.mapping)
            train_samples += data.mask_duplicate_positive_train.shape[1]
            test_samples += data.mask_duplicate_positive_test.shape[1]
            val_samples += data.mask_duplicate_positive_val.shape[1]
            total_nodes += data.num_nodes
            total_edges += data.num_edges

            data.mask_duplicate_positive = np.concatenate(
                (data.mask_duplicate_positive_train, data.mask_duplicate_positive_test,
                 data.mask_duplicate_positive_val),
                axis=-1)

            if inference is False:
                # resample negative duplication samples
                get_duplicate_mask(data, resplit=False, use_jaccard_ranking=True)
            else:
                data.mask_duplicate_negative_train = convert_mask_from_node_names(duplicate_train_neg_list[i],
                                                                                  data.mapping)
                data.mask_duplicate_negative_val = convert_mask_from_node_names(duplicate_val_neg_list[i],
                                                                                data.mapping)
                data.mask_duplicate_negative_test = convert_mask_from_node_names(duplicate_test_neg_list[i],
                                                                                 data.mapping)

            dists = precompute_dist_data(data.edge_index.numpy(), data.num_nodes, approximate=args.approximate)
            print_progress_bar(i + 1, len(dataset), prefix="subgraphs: ", length=50)

            # Creates a Tensor from a numpy.ndarray.
            data.dists = torch.from_numpy(dists).float()

            data.duplicate_index = torch.from_numpy(
                np.concatenate((data.mask_duplicate_positive_train,
                                data.mask_duplicate_positive_test,
                                data.mask_duplicate_positive_val),
                               axis=1)).long()

            if remove_feature:
                data.x = torch.ones((data.x.shape[0], 1))

            data_list.append(data)

        print()
        print("=============================")
        print("total positive samples:", train_samples + test_samples + val_samples)
        print("=> train:", train_samples)
        print("=> test:", test_samples)
        print("=> val", val_samples)
        print("total nodes:", total_nodes)
        print("total edges", total_edges)
        print()
    return data_list


def nx_to_tg_data(graphs, features, orig_features, duplicate_node_mappings, edge_labels=None):
    """
    Function for converting raw data to Data object.
    :param graphs: array including the networkx graph object for each subgraph
    :param features: array including the feature vectors for each node in each subgraph
    :param orig_features: array including the original feature vectors for each node in each subgraph
    :param duplicate_node_mappings: array including the mapping between duplicate nodes in each subgraph
    :param edge_labels: array including the edge labels between nodes in each subgraph
    :return: data_list array for each subgraph with prepared data
    """
    data_list = []

    print("processing ", len(graphs), " graphs")
    for i in range(len(graphs)):
        feature = features[i]
        orig_feature = orig_features[i]

        graph = graphs[i].copy()
        graph.remove_edges_from(graph.selfloop_edges())

        # create mapping for relabeling graph nodes
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))

        # relabel and sum up mapping between duplicate nodes
        graph_duplicate_node_mappings = []
        for key, value in mapping.items():
            if key in duplicate_node_mappings.keys():
                if duplicate_node_mappings[key] in mapping.keys():
                    graph_duplicate_node_mappings.append((value, mapping[duplicate_node_mappings[key]]))

        nx.relabel_nodes(graph, mapping, copy=False)

        x = np.zeros(feature.shape)
        graph_nodes = list(graph.nodes)

        # get feature vector x for each node in subgraph
        for m in range(feature.shape[0]):
            x[graph_nodes[m]] = feature[m]
        x = torch.from_numpy(x).float()

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
        data = Data(x=x, edge_index=edge_index)
        data.graph = graph

        # get edge_labels
        if edge_labels[0] is not None:
            edge_label = edge_labels[i]
            data.mask_duplicate_positive = None
            data.mapping = mapping
            data.feature = feature
            data.orig_feature = orig_feature
            data.graph_duplicate_node_mappings = graph_duplicate_node_mappings
            data.duplicate_index = duplicate_index
        data_list.append(data)

    return data_list


def load_graphs(dataset_str, dataset_duplicates_str):
    """
    Main data load function from relevant files.
    :param dataset_str: name of dataset in folder "../dataprep/data/"
    :param dataset_duplicates_str: file name of related ".map" file in "../dataprep/generated/"
    :return: separated arrays including different data for each subgraph
    """
    node_labels = [None]
    edge_labels = [None]
    idx_train = [None]
    idx_val = [None]
    idx_test = [None]
    orig_features = [None]

    dataset_dir = '../dataprep/data/' + dataset_str
    print("Loading data...")

    G = json_graph.node_link_graph(json.load(open(dataset_dir + "-G.json")))

    edge_labels_internal = json.load(open(dataset_dir + "-class_map.json"))
    edge_labels_internal = {i: l for i, l in edge_labels_internal.items()}

    # loading duplicate mappings
    with open("../dataprep/generated/" + dataset_duplicates_str, 'rb') as file:
        duplicate_node_mappings = pickle.load(file)

    train_ids = [n for n in G.nodes()]
    train_labels = np.array([edge_labels_internal[i] for i in train_ids])

    if train_labels.ndim == 1:
        train_labels = np.expand_dims(train_labels, 1)
    edge_labels = train_labels

    print("Using only features..")
    feats = np.load(dataset_dir + "-feats.npy")
    orig_feats = feats

    # Logistic gets thrown off by big counts, so log transform num comments and score
    feats[:, 0] = np.log(feats[:, 0] + 1.0)
    feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))

    feat_id_map = json.load(open(dataset_dir + "-id_map.json"))
    feat_id_map = {id: val for id, val in feat_id_map.items()}

    train_feats = feats[[feat_id_map[id] for id in train_ids]]
    orig_train_feats = orig_feats[[feat_id_map[id] for id in train_ids]]

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

    print("Creating features")
    features = [train_feats[id_temp, :] + 0.1 for id_temp in id_all]
    orig_features = [orig_train_feats[id_temp, :] for id_temp in id_all]

    return graphs, features, orig_features, edge_labels, dict(
        duplicate_node_mappings), node_labels, idx_train, idx_val, idx_test


def load_graphs_v2(dataset_str, dataset_duplicates_str, selected_features=[]):
    """
    Main data load function from relevant files by using more advanced features. (v2)
    :param selected_features: list of attributes, which should be used for feature vector generation
    :param dataset_str: name of dataset in folder "../dataprep/data/"
    :param dataset_duplicates_str: file name of related ".map" file in "../dataprep/generated/"
    :return: separated arrays including different data for each subgraph
    """
    node_labels = [None]
    edge_labels = [None]
    idx_train = [None]
    idx_val = [None]
    idx_test = [None]
    orig_features = [None]

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

    feats = None

    if len(selected_features) == 0:
        feats = np.ones((attributes_raw_features[list(attributes_raw_features.keys())[0]].shape[0], 1))

    for attribute_name, attribute_feature_array in attributes_raw_features.items():
        if attribute_name in selected_features:
            if feats is None:
                feats = attribute_feature_array
            else:
                feats = np.hstack((feats, attribute_feature_array))

    orig_feats = feats

    feat_id_map = json.load(open(dataset_dir + "-id_map.json"))
    feat_id_map = {id: val for id, val in feat_id_map.items()}

    train_feats = feats[[feat_id_map[id] for id in train_ids]]
    orig_train_feats = orig_feats[[feat_id_map[id] for id in train_ids]]

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

    print("Creating features")
    features = [train_feats[id_temp, :] for id_temp in id_all]
    orig_features = [orig_train_feats[id_temp, :] for id_temp in id_all]

    return graphs, features, orig_features, edge_labels, dict(
        duplicate_node_mappings), node_labels, idx_train, idx_val, idx_test


def load_tg_dataset(name='udbms', dataset_duplicates_str="person_duplicate", version='v2', selected_features=[]):
    """
    Loading the graph from dataset and convert it to expected format.
    :param version: 'v1': feature labeling, 'v2': advanced feature extraction using 1-Hot-Enc., geo data, ...
    :param name: name of dataset in folder "../dataprep/data/"
    :param dataset_duplicates_str: file name of related ".map" file in "../dataprep/generated/"
    :return: data list for each subgraph calculated by nx_to_tg_data method
    """
    print("Load tg dataset")

    if version == 'v1':
        graphs, features, orig_features, \
        edge_labels, duplicate_node_mappings, _, _, _, _ = load_graphs(name, dataset_duplicates_str)
    elif version == 'v2':
        graphs, features, orig_features, edge_labels, \
        duplicate_node_mappings, _, _, _, _ = load_graphs_v2(name, dataset_duplicates_str, selected_features)

    return nx_to_tg_data(graphs, features, orig_features, duplicate_node_mappings, edge_labels)
