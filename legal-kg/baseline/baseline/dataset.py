import json
import os.path
import pickle

import networkx as nx
import numpy as np
import torch
import torch.utils.data
from networkx.readwrite import json_graph
from torch_geometric.data import Data
from utils import get_duplicate_mask, convert_mask_from_node_names


def get_tg_dataset(dataset_name, dataset_duplicates, inference=False, selected_features=[]):
    """
    Loading the dataset and prepare it for later use.
    :param dataset_duplicates:
    :param selected_features: labels of feature attributes to be included in the feature vectors
    :param inference: whether is in inference mode or nor (loading same cached negative samples for evaluation)
    :param dataset_name: name of dataset in folder "../dataprep/data/"
    :return: data list array for each subgraph
    """
    try:
        dataset = load_tg_dataset(dataset_name, dataset_duplicates, selected_features=selected_features)
    except:
        raise NotImplementedError

    # creating dataset caching directory if required
    if not os.path.isdir('../pgnn/datasets/cache'):
        os.mkdir('../pgnn/datasets/cache')
    f1_name = '../pgnn/datasets/cache/' + dataset_name + '-1_duplicates_train.dat'
    f2_name = '../pgnn/datasets/cache/' + dataset_name + '-1_duplicates_val.dat'
    f3_name = '../pgnn/datasets/cache/' + dataset_name + '-1_duplicates_test.dat'

    f4_name = '../pgnn/datasets/cache/' + dataset_name + '-1_duplicates_train_neg.dat'
    f5_name = '../pgnn/datasets/cache/' + dataset_name + '-1_duplicates_val_neg.dat'
    f6_name = '../pgnn/datasets/cache/' + dataset_name + '-1_duplicates_test_neg.dat'

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

    for i, data in enumerate(dataset):
        # convert node names to temporary node ids in subgraph (because node ids seem to be random every time)
        data.mask_duplicate_positive_train = convert_mask_from_node_names(duplicate_train_list[i], data.mapping)
        data.mask_duplicate_positive_val = convert_mask_from_node_names(duplicate_val_list[i], data.mapping)
        data.mask_duplicate_positive_test = convert_mask_from_node_names(duplicate_test_list[i], data.mapping)

        data.mask_duplicate_positive = np.concatenate(
            (data.mask_duplicate_positive_train, data.mask_duplicate_positive_test, data.mask_duplicate_positive_val),
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

        data.duplicate_index = torch.from_numpy(
            np.concatenate((data.mask_duplicate_positive_train,
                            data.mask_duplicate_positive_test,
                            data.mask_duplicate_positive_val),
                           axis=1)).long()

        data_list.append(data)

    return data_list


def nx_to_tg_data(graphs, features, duplicate_node_mappings, feature_shapes, edge_labels=None):
    """
    Function for converting raw data to Data object.
    :param feature_shapes: length of separate feature vectors
    :param graphs: array including the networkx graph object for each subgraph
    :param features: array including the feature vectors for each node in each subgraph
    :param duplicate_node_mappings: array including the mapping between duplicate nodes in each subgraph
    :param edge_labels: array including the edge labels between nodes in each subgraph
    :return: data_list array for each subgraph with prepared data
    """
    data_list = []

    print("processing ", len(graphs), " graphs")
    for i in range(len(graphs)):
        feature = features[i]

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

        # get duplicate index => duplicating each existent duplicate from one side to other side (e.g. 8 => 1 && 8 <= 1)
        duplicate_index = None
        try:
            duplicate_index = np.array(list(graph_duplicate_node_mappings))
            duplicate_index = np.concatenate((duplicate_index, duplicate_index[:, ::-1]), axis=0)
            duplicate_index = torch.from_numpy(duplicate_index).long().permute(1, 0)
        except:
            duplicate_index = torch.from_numpy(duplicate_index).long()

        # create Data object from torch_geometric, including all data for subgraph
        data = Data(x=x)
        data.graph = graph

        # get edge_labels
        if edge_labels[0] is not None:
            data.mapping = mapping
            data.feature = feature
            data.graph_duplicate_node_mappings = graph_duplicate_node_mappings
            data.duplicate_index = duplicate_index
            data.feature_shapes = feature_shapes
        data_list.append(data)

    return data_list


def load_graphs(dataset_str, dataset_duplicates_str, selected_features=[]):
    """
    Main data load function from relevant files by using more advanced features. (v2)
    :param selected_features: list of attributes, which should be used for feature vector generation
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

    feats = None

    feature_shapes = []

    for attribute_name, attribute_feature_array in attributes_raw_features.items():
        if attribute_name in selected_features:
            if feats is None:
                feats = attribute_feature_array
            else:
                feats = np.hstack((feats, attribute_feature_array))

            feature_shapes.append(attribute_feature_array.shape[1])

    feat_id_map = json.load(open(dataset_dir + "-id_map.json"))
    feat_id_map = {id: val for id, val in feat_id_map.items()}

    train_feats = feats[[feat_id_map[id] for id in train_ids]]

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

    return graphs, features, edge_labels, dict(duplicate_node_mappings), feature_shapes


def load_tg_dataset(name='udbms', dataset_duplicates_str="person_duplicate", selected_features=[]):
    """
    Loading the graph from dataset and convert it to expected format.
    :param name: name of dataset in folder "../dataprep/data/"
    :param dataset_duplicates_str: file name of related ".map" file in "../dataprep/generated/"
    :return: data list for each subgraph calculated by nx_to_tg_data method
    """
    print("Load tg dataset")

    graphs, features, edge_labels, duplicate_node_mappings, feature_shapes = \
        load_graphs(name, dataset_duplicates_str, selected_features)

    return nx_to_tg_data(graphs, features, duplicate_node_mappings, feature_shapes, edge_labels)
