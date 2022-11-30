import torch
import networkx as nx
import numpy as np
import multiprocessing as mp
import random


def get_duplicate_mask(data, remove_ratio=0.2, resplit=True, neg_samples_ratio=1, use_jaccard_ranking=False):
    """
    Function for creating the sample sets for training, evaluation and testing.
    :param use_jaccard_ranking: whether use the jaccard coefficient for ranking negative samples on TRAINING SET
    :param data: data object of subgraph
    :param remove_ratio: ratio for splitting up sample sets into train/test/val sets
    :param resplit: boolean whether the positive samples should be generated again or not
    :param neg_samples_ratio: float, how many negative samples should be generated (default: 1)
    """
    if resplit:
        data.mask_duplicate_positive = data.duplicate_index.numpy()
        data.mask_duplicate_positive_train, data.mask_duplicate_positive_val, data.mask_duplicate_positive_test = \
            split_duplicates(data.mask_duplicate_positive, remove_ratio)

    resample_mask_duplicate_negative(data, neg_samples_ratio, use_jaccard_ranking)


def split_duplicates(duplicates, remove_ratio):
    """
    Function for creating the sample sets by splitting up the input data.
    :param duplicates: object with all duplicate mappings in subgraph
    :param remove_ratio:  ratio for splitting up sample sets into train/test/val sets
    :return: train/test/val sets containing duplicates
    """
    if len(duplicates.shape) > 1:
        e = duplicates.shape[1]
        duplicates = duplicates[:, np.random.permutation(e)]

        split1 = int((1 - remove_ratio) * e)
        split2 = int((1 - remove_ratio / 2) * e)

        duplicates_train = duplicates[:, :split1]
        duplicates_val = duplicates[:, split1:split2]
        duplicates_test = duplicates[:, split2:]
    else:
        duplicates_train = np.asarray([[], []])
        duplicates_val = np.asarray([[], []])
        duplicates_test = np.asarray([[], []])

    return duplicates_train, duplicates_val, duplicates_test


def resample_mask_duplicate_negative(data, neg_samples_ratio, use_jaccard_ranking):
    """
    Function for setting negative duplication sets in data object.
    :param use_jaccard_ranking: whether use the jaccard coefficient for ranking negative samples on TRAINING SET
    :param data: data object of subgraph
    :param neg_samples_ratio: float, how many negative samples should be generated
    """
    data.mask_duplicate_negative_train = get_mask_duplicate_negative(data.mask_duplicate_positive_train,
                                                                     data.graph,
                                                                     use_jaccard_ranking=use_jaccard_ranking,
                                                                     num_nodes=data.num_nodes,
                                                                     num_negtive_duplicates=
                                                                     int(data.mask_duplicate_positive_train.shape[
                                                                         1] * neg_samples_ratio))
    data.mask_duplicate_negative_val = get_mask_duplicate_negative(data.mask_duplicate_positive_val,
                                                                   data.graph,
                                                                   use_jaccard_ranking=False,
                                                                   num_nodes=data.num_nodes,
                                                                   num_negtive_duplicates=
                                                                   int(data.mask_duplicate_positive_val.shape[
                                                                       1] * neg_samples_ratio))
    data.mask_duplicate_negative_test = get_mask_duplicate_negative(data.mask_duplicate_positive_test,
                                                                    data.graph,
                                                                    use_jaccard_ranking=False,
                                                                    num_nodes=data.num_nodes,
                                                                    num_negtive_duplicates=
                                                                    int(data.mask_duplicate_positive_test.shape[
                                                                        1] * neg_samples_ratio))


def get_mask_duplicate_negative(mask_duplicate_positive, graph, use_jaccard_ranking, num_nodes, num_negtive_duplicates):
    """
    Function for creating negative duplicate samples based on the positive sample set.
    :param use_jaccard_ranking: whether use the jaccard coefficient for ranking negative samples or random selection
    :param graph: networkx (sub)graph as input for jaccard calculation
    :param mask_duplicate_positive: positive sample set, for which the negative sample set should be generated
    :param num_nodes: number of nodes in subgraph
    :param num_negtive_duplicates: how many negative duplicate should be generated
    :return: negative sample set
    """
    mask_duplicate_positive_set = []
    for i in range(mask_duplicate_positive.shape[1]):
        mask_duplicate_positive_set.append(tuple(mask_duplicate_positive[:, i]))
    mask_duplicate_positive_set = set(mask_duplicate_positive_set)

    negative_duplicates_count = num_negtive_duplicates

    # if using jaccard for ranking negative samples, 3 times more samples should be generated
    if use_jaccard_ranking is True:
        negative_duplicates_count *= 3
        jaccard_scores = np.zeros((negative_duplicates_count, ), dtype=np.float)

    mask_duplicate_negative = np.zeros((2, negative_duplicates_count), dtype=mask_duplicate_positive.dtype)
    for i in range(negative_duplicates_count):
        while True:
            mask_temp = tuple(np.random.choice(num_nodes, size=(2,), replace=False))
            if mask_temp not in mask_duplicate_positive_set:
                mask_duplicate_negative[:, i] = mask_temp

                # calculate jaccard coefficients between negative pair
                if use_jaccard_ranking is True:
                    jc = nx.jaccard_coefficient(graph, [(mask_temp[0], mask_temp[1])])
                    for u, v, p in jc:
                        jaccard_scores[i] = p

                break

    # get indices of max jaccard scores between pairs (in the amount of positive sample count)
    if use_jaccard_ranking is True:
        max_idx = (-jaccard_scores).argsort()[:num_negtive_duplicates]
        # return duplicate mask with highest ranked pairs
        return mask_duplicate_negative[:, max_idx]

    return mask_duplicate_negative


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    """
    :param graph: networkx subgraph object
    :param node_range: array with all nodes, from which the path lengths should be calculated
    :param cutoff: maximum path length for neighbor lookup
    :return: dictionary with path lengths to all neighbors for each node in node_range
    """
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def all_pairs_shortest_path_length_parallel(graph, cutoff=None, num_workers=70):
    """
    Function for calculating path lengths in parallel.
    :param graph: networkx subgraph object
    :param cutoff: maximum path length for neighbor lookup
    :param num_workers: integer as count of parallel workers
    :return: dictionary with path lengths to all neighbors for each node
    """
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
                                args=(
                                    graph,
                                    nodes[int(len(nodes) / num_workers * i):int(len(nodes) / num_workers * (i + 1))],
                                    cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, num_nodes, approximate=0):
    """
    Function for pre-computing the distance data for the subgraph based on the edge_index.
    Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
    :param edge_index: object with all edges in subgraph
    :param num_nodes: number of nodes in subgraph (n)
    :param approximate: maximum path length for neighbor lookup (default: 0 => no approximation)
    :return: matrix with distances between all nodes to each other, with n*n shape
    """
    graph = nx.Graph()
    edge_list = edge_index.transpose(1, 0).tolist()

    graph.add_edges_from(edge_list)

    n = num_nodes
    dists_array = np.zeros((n, n))
    dists_dict = all_pairs_shortest_path_length_parallel(graph, cutoff=approximate if approximate > 0 else None)

    for i, node_i in enumerate(graph.nodes()):
        if dists_dict.get(node_i) is None:
            continue

        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(graph.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist != -1:
                # dists_array[i, j] = 1 / (dist + 1)
                dists_array[node_i, node_j] = 1 / (dist + 1)

    return dists_array


def get_random_anchorset(n, c=0.5):
    """
    Function for creating array with random anchsor sets based on the number of nodes.
    In our task: prevent nodes involved in duplication to be part of an anchor set
    :param n: number of nodes in subgraph
    :param c: factor for anchor set size calculation (default: 0.5)
    :return: array with random anchor sets of different size
    """

    m = int(np.log2(n))
    copy = int(c * m)

    anchorset_id = []
    for i in range(m):
        anchor_size = int(n / np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n, size=anchor_size, replace=False))
    return anchorset_id


def get_max(dist_temp, device):
    dist_max = torch.zeros((1, dist_temp.shape[0])).to(device)
    dist_argmax = torch.zeros((1, dist_temp.shape[0])).long().to(device)

    for i in range(dist_temp.shape[0]):
        max_value = 0
        max_index = 0
        second_max_value = 0
        second_max_index = 0

        for j in range(dist_temp.shape[1]):
            value = float(dist_temp[i, j])

            if value > max_value:
                second_max_value = max_value
                second_max_index = max_index
                max_value = value
                max_index = j
            elif value > second_max_value:
                second_max_value = value
                second_max_index = j

        if max_value == 1:
            dist_max[0, i] = second_max_value
            dist_argmax[0, i] = second_max_index
        else:
            dist_max[0, i] = max_value
            dist_argmax[0, i] = max_index

    return dist_max, dist_argmax


def get_dist_max(anchorset_id, dist, device, data):
    """
    Function for calculating the maximum distance to the nodes in each anchor set.
    :param data:
    :param anchorset_id: array with random anchor sets of different size
    :param dist: pre-computed distance matrix of subgraph
    :param device: torch device for computing
    :return: max distances within each anchor set, indices of node with max distance
    """
    dist_max = torch.zeros((dist.shape[0], len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0], len(anchorset_id))).long().to(device)

    for i in range(len(anchorset_id)):
        temp_id = anchorset_id[i]
        dist_temp = dist[:, temp_id]

        # print(dist)
        # print("")
        # print("------")
        # print(data.duplicate_index)
        # print(temp_id)
        # print("")
        # print(dist_temp)
        dist_max_temp, dist_argmax_temp = get_max(dist_temp, device)

        dist_max[:, i] = dist_max_temp
        dist_argmax[:, i] = dist_argmax_temp

    return dist_max, dist_argmax


def preselect_anchor(data, layer_num=2, anchor_num=32, anchor_size_num=4, device='cpu'):
    """
    Function for randomly selecting anchor sets and calculating required distances.
    :param data: data object of subgraph
    :param layer_num: count of layers used
    :param anchor_num: count of anchors to be used
    :param anchor_size_num: size of anchors
    :param device: torch device for computing
    """
    # not used
    # data.anchor_size_num = anchor_size_num
    # data.anchor_set = []
    # anchor_num_per_size = anchor_num // anchor_size_num
    # for i in range(anchor_size_num):
    #     anchor_size = 2 ** (i + 1) - 1
    #
    #     anchors = np.random.choice(data.num_nodes, size=(layer_num, anchor_num_per_size, anchor_size), replace=True)
    #     data.anchor_set.append(anchors)
    # not used

    anchorset_id = get_random_anchorset(data.num_nodes, c=1)

    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device, data)


def convert_mask_to_node_names(mask, mapping):
    """
    :param mask: mask with mapping between duplicates by using the node ids
    :param mapping: mapping for all nodes in subgraph between node id and their unique name in dataset
    :return: mask with mapping between duplicates by using the unique node names
    """
    array = np.empty(mask.shape, dtype=np.dtype('U256'))

    for dup_index in range(mask.shape[1]):
        first = mask[0][dup_index]
        second = mask[1][dup_index]

        for key, value in mapping.items():
            if value == first:
                array[0][dup_index] = key
            if value == second:
                array[1][dup_index] = key

    return array


def convert_mask_from_node_names(mask, mapping):
    """
    :param mask: mask with mapping between duplicates by using the unique node names
    :param mapping: mapping for all nodes in subgraph between node id and their unique name in dataset
    :return: mask with mapping between duplicates by using the node ids
    """
    array = np.ndarray(mask.shape, dtype=int)

    for dup_name in range(mask.shape[1]):
        first = mask[0][dup_name]
        second = mask[1][dup_name]

        for key, value in mapping.items():
            if key == first:
                array[0][dup_name] = value
            if key == second:
                array[1][dup_name] = value

    return array


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
