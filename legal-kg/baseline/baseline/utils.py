import networkx as nx
import numpy as np

from sklearn.utils import check_consistent_length, assert_all_finite, column_or_1d


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
                                                                     data.mask_duplicate_positive,
                                                                     use_jaccard_ranking=use_jaccard_ranking,
                                                                     num_nodes=data.num_nodes,
                                                                     num_negative_duplicates=
                                                                     int(data.mask_duplicate_positive_train.shape[
                                                                             1] * neg_samples_ratio))
    data.mask_duplicate_negative_val = get_mask_duplicate_negative(data.mask_duplicate_positive_val,
                                                                   data.graph,
                                                                   data.mask_duplicate_positive,
                                                                   use_jaccard_ranking=False,
                                                                   num_nodes=data.num_nodes,
                                                                   num_negative_duplicates=
                                                                   int(data.mask_duplicate_positive_val.shape[
                                                                           1] * neg_samples_ratio))
    data.mask_duplicate_negative_test = get_mask_duplicate_negative(data.mask_duplicate_positive_test,
                                                                    data.graph,
                                                                    data.mask_duplicate_positive,
                                                                    use_jaccard_ranking=False,
                                                                    num_nodes=data.num_nodes,
                                                                    num_negative_duplicates=
                                                                    int(data.mask_duplicate_positive_test.shape[
                                                                            1] * neg_samples_ratio))


def get_mask_duplicate_negative(mask_duplicate_positive, graph, duplicate_mask, use_jaccard_ranking,
                                num_nodes, num_negative_duplicates):
    """
    Function for creating negative duplicate samples based on the positive sample set.
    :param duplicate_mask: all positive duplicate samples
    :param use_jaccard_ranking: whether use the jaccard coefficient for ranking negative samples or random selection
    :param graph: networkx (sub)graph as input for jaccard calculation
    :param mask_duplicate_positive: positive sample set, for which the negative sample set should be generated
    :param num_nodes: number of nodes in subgraph
    :param num_negative_duplicates: how many negative duplicate should be generated
    :return: negative sample set
    """
    mask_duplicate_positive_set = []
    for i in range(duplicate_mask.shape[1]):
        mask_duplicate_positive_set.append(tuple(duplicate_mask[:, i]))
    mask_duplicate_positive_set = set(mask_duplicate_positive_set)

    negative_duplicates_count = num_negative_duplicates

    # if using jaccard for ranking negative samples, 3 times more samples should be generated
    if use_jaccard_ranking is True:
        negative_duplicates_count *= 3
        jaccard_scores = np.zeros((negative_duplicates_count,), dtype=np.float)

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
        max_idx = (-jaccard_scores).argsort()[:num_negative_duplicates]
        # return duplicate mask with highest ranked pairs
        return mask_duplicate_negative[:, max_idx]

    return mask_duplicate_negative


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


def _binary_clf_curve(y_true, y_score, threshold_step):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    threshold_step : float
    Returns
    -------
    fps : array, shape = [n_thresholds]
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
    tns : array, shape = [n_thresholds <= len(np.unique(y_score))]
    fns : array, shape = [n_thresholds <= len(np.unique(y_score))]
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    pos_label = 1

    check_consistent_length(y_true, y_score, None)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    thresholds = np.arange(start=0.2, stop=0.9, step=threshold_step)

    tps = np.asarray([])
    fps = np.asarray([])
    tns = np.asarray([])
    fns = np.asarray([])

    for i in range(len(thresholds)):
        threshold = thresholds[i]
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for j in range(len(y_true)):
            score = y_score[j]
            is_positive = y_true[j]

            if score > threshold:
                if is_positive:
                    tp += 1
                else:
                    fp += 1
            else:
                if is_positive:
                    fn += 1
                else:
                    tn += 1

        tps = np.append(tps, tp)
        fps = np.append(fps, fp)
        tns = np.append(tns, tn)
        fns = np.append(fns, fn)

    return fps, tps, tns, fns, thresholds


def precision_recall_curve(y_true, probas_pred, threshold_step=0.005):
    """Compute precision-recall pairs for different probability thresholds
    Note: this implementation is restricted to the binary classification task.
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold.  This ensures that the graph starts on the
    y axis.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.
    probas_pred : array, shape = [n_samples]
        Estimated probabilities or decision function.
    threshold_step : float
    Returns
    -------
    precision : array, shape = [n_thresholds + 1]
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.
    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    """
    fps, tps, tns, fns, thresholds = _binary_clf_curve(y_true, probas_pred, threshold_step)
    np.seterr(divide='ignore', invalid='ignore')

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0

    recall = tps / (tps + fns)

    return precision, recall, thresholds
