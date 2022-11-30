# %%
import json
import networkx as nx
import numpy as np
import os
import pickle
import random
import multiprocessing as mp


def makeLabel(node1, node2):
    if node1 < node2:
        return node1 + "_" + node2
    return node2 + "_" + node1


# %%
class UDBMSDataset:
    G = nx.Graph()
    id_map = {}
    nodes_list = []
    graph_map = {}
    class_map = {}
    links_list = []
    linked_nodes_map = {}
    nodes_attributes_map = {}
    features_map = {}
    id_to_features_map = {}

    def create_features_list(self, node_id, node_attributes_map):
        features_list = []
        for attribute, feature_map in self.features_map.items():
            node_attribute_value = node_attributes_map.get(attribute)
            if node_attribute_value is None:
                features_list.append(0)
            else:
                if feature_map.get(node_attribute_value) is None:
                    # this is an error
                    features_list.append(0)
                else:
                    features_list.append(feature_map.get(node_attribute_value))
        return features_list

    # we're going to read a file from below url
    # https://www.helsinki.fi/en/researchgroups/unified-database-management-systems-udbms/datasets/person-dataset
    def create_nodes(self, input_file_name):
        f = open(input_file_name, "r")
        lines = f.readlines()
        f.close()

        id_count = 0
        # we're going to read a file which has tab separated triples
        for line in lines:
            line = line.strip()
            tup = tuple(line.split("\t"))
            if len(tup) != 3:
                print("Each line must be a triple. Line was: ", line)
                continue

            # first element of the tuple is almost always an entity
            if self.id_map.get(tup[0]) is None:
                self.id_map[tup[0]] = id_count
                id_count += 1
                self.G.add_node(tup[0])
                self.class_map[tup[0]] = [1, 0]
                node_map = {}
                node_map["test"] = False
                node_map["val"] = False
                node_map["id"] = tup[0]
                node_attributes_map = self.nodes_attributes_map.get(tup[0])
                features_list = None
                if node_attributes_map is not None and len(node_attributes_map) > 0:
                    features_list = self.create_features_list(tup[0], node_attributes_map)
                else:
                    features_list = [0] * len(self.features_map)
                node_map["features"] = features_list
                self.id_to_features_map[tup[0]] = features_list
                self.nodes_list.append(node_map)
            if tup[1] == 'subject':
                continue
            relation = ""
            attribute = ""
            rel_list = ['relation', 'relative', 'spouse', 'child', 'parent', 'partner', 'predecessor', 'successor',
                        'opponent', 'rival']
            if tup[1] in rel_list:
                relation = tup[1]
            else:
                attribute = tup[1]
            if relation != "":
                if self.id_map.get(tup[2]) is None:
                    self.id_map[tup[2]] = id_count
                    id_count += 1
                    self.G.add_node(tup[2])
                    self.class_map[tup[2]] = [1, 0]
                    node_map = {}
                    node_map["test"] = False
                    node_map["val"] = False
                    if id_count % 10 == 8:
                        node_map["val"] = True
                    if id_count % 10 == 9:
                        node_map["test"] = True
                    node_map["id"] = tup[2]
                    node_attributes_map = self.nodes_attributes_map.get(tup[2])
                    features_list = None
                    if node_attributes_map is not None and len(node_attributes_map) > 0:
                        features_list = self.create_features_list(tup[2], node_attributes_map)
                    else:
                        features_list = [0] * len(self.features_map)
                    node_map["features"] = features_list
                    self.id_to_features_map[tup[2]] = features_list
                    self.nodes_list.append(node_map)

    def prepare_node_attributes(self, input_file_name, attributes_file_name):
        f = open(attributes_file_name, "r")
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            self.features_map[line] = {}
        f.close()

        f = open(input_file_name, "r")
        lines = f.readlines()
        f.close()

        id_count = 0
        # we're going to read a file which has tab separated triples
        for line in lines:
            line = line.strip()
            tup = tuple(line.split("\t"))
            if len(tup) != 3:
                print("Each line must be a triple. Line was: ", line)
                continue

            # first element of the tuple is almost always an entity
            if self.nodes_attributes_map.get(tup[0]) is None:
                self.nodes_attributes_map[tup[0]] = {}
            if tup[1] == 'subject':
                continue
            relation = ""
            attribute = ""
            rel_list = ['relation', 'relative', 'spouse', 'child', 'parent', 'partner', 'predecessor', 'successor',
                        'opponent', 'rival']
            if tup[1] in rel_list:
                relation = tup[1]
            else:
                attribute = tup[1]
            if attribute != "":
                node_attributes_map = self.nodes_attributes_map.get(tup[0])
                if node_attributes_map.get(attribute) is None:
                    # get feature map for attribute
                    feature_map = self.features_map.get(attribute)
                    if feature_map.get(tup[2]) is None:
                        # the length of this map + 1 is the new ordinal id we create for that attribute value
                        feature_map[tup[2]] = len(feature_map) + 1
                        self.features_map[attribute] = feature_map
                    node_attributes_map[attribute] = tup[2]
                    self.nodes_attributes_map[tup[0]] = node_attributes_map

    def add_edges(self, input_file_name):
        f = open(input_file_name, "r")
        lines = f.readlines()
        f.close()

        # we're going to read a file which has tab separated triples
        for line in lines:
            line = line.strip()
            tup = tuple(line.split("\t"))
            if len(tup) != 3:
                print("Each line must be a triple. Line was: ", line)
                continue
            # we want to find node properties to add
            if tup[1] == 'subject':
                continue
            relation = ""
            attribute = ""
            rel_list = ['relation', 'relative', 'spouse', 'child', 'parent', 'partner', 'predecessor', 'successor',
                        'opponent', 'rival']
            if tup[1] in rel_list:
                relation = tup[1]
            else:
                attribute = tup[1]
            if relation != "":
                linked_nodes = makeLabel(tup[0], tup[2])
                if self.linked_nodes_map.get(linked_nodes) is None:
                    link_map = {}
                    link_map["source"] = tup[0]
                    link_map["target"] = tup[2]
                    # link_map["relation_type"] = tup[1]
                    self.links_list.append(link_map)
                    self.G.add_edge(tup[0], tup[2])
                    self.linked_nodes_map[linked_nodes] = tup[1]

        # loading duplicate mappings
        with open("./generated/person_duplicate_without_subject_feat_25percent.map", 'rb') as file:
            duplicate_node_mappings = pickle.load(file)

        use_cache = False
        if use_cache:
            with open("./generated/person_duplicate_without_subject_feat.dist", 'rb') as file:
                dist = pickle.load(file)
        else:
            dist = precompute_dist_data(self.G, 0, approximate=4)
            with open("./generated/person_duplicate_without_subject_feat.dist", 'wb') as file:
                pickle.dump(dist, file)

        count = 0
        for duplicates in duplicate_node_mappings:
            original = duplicates[0]
            duplicate = duplicates[1]

            neighbor_dist_original = dist.get(original)
            neighbor_dist_duplicate = dist.get(duplicate)

            if neighbor_dist_original is None or neighbor_dist_duplicate is None:
                continue

            direct_neighbors = []
            for key, value in neighbor_dist_original.items():
                if value == 1:
                    direct_neighbors.append(key)

            for neighbor in direct_neighbors:
                for key, value in neighbor_dist_duplicate.items():
                    if key == neighbor and value != 1:
                        # add missing edge
                        print("edge missing between ", duplicate, " and ", neighbor)
                        count += 1
                        linked_nodes = makeLabel(duplicate, neighbor)
                        if self.linked_nodes_map.get(linked_nodes) is None:
                            link_map = {"source": duplicate, "target": neighbor}
                            self.links_list.append(link_map)
                            self.G.add_edge(duplicate, neighbor)

        print("created missing edges: ", count)

    def save_id_map(self, prefix):
        filename = prefix + "udbms-id_map.json"
        with open(filename, 'w') as f:
            json.dump(self.id_map, f)

    def save_class_map(self, prefix):
        filename = prefix + "udbms-class_map.json"
        with open(filename, 'w') as f:
            json.dump(self.class_map, f)

    def save_features_map_legend(self, prefix):
        json_data = self.features_map
        features_map_legend_file = prefix + "udbms-features-map-legend.json"
        with open(features_map_legend_file, 'w') as f:
            json.dump(json_data, f)

    def save_features(self, prefix, shape_tuple):
        # features = np.zeros(shape=shape_tuple)
        # np.save(prefix+'udbms-feats.npy', features)
        features_list_of_list = []
        counter = 0

        for k, v in sorted(self.id_map.items(), key=lambda x: x[1]):
            if sum(self.id_to_features_map.get(k)) == 0:
                counter += 1
                print(k)

            features_list_of_list.append(self.id_to_features_map.get(k))

        print(counter)
        print(len(self.id_to_features_map))
        features_narray = np.array(features_list_of_list)
        np.save(prefix + 'udbms-feats.npy', features_narray)

    def save_graph(self, prefix):
        print("Number of edges: ", self.G.number_of_edges())
        print("Number of nodes: ", self.G.number_of_nodes())
        filename = prefix + "udbms-G.json"
        output_map = {}
        output_map["directed"] = False
        meta_data_map = {}
        meta_data_map["name"] = "udbms"
        output_map["graph"] = meta_data_map
        output_map["nodes"] = self.nodes_list
        output_map["links"] = self.links_list
        output_map["multigraph"] = False
        with open(filename, 'w') as f:
            json.dump(output_map, f)

    def main(self, prefix, input_file_name, attributes_file_name):
        self.prepare_node_attributes(input_file_name, attributes_file_name)
        self.create_nodes(input_file_name)
        self.add_edges(input_file_name)
        self.save_graph(prefix)
        self.save_id_map(prefix)
        self.save_class_map(prefix)
        shape_tuple = (len(self.id_map), 2)
        self.save_features(prefix, shape_tuple)
        self.save_features_map_legend(prefix)

def single_source_shortest_path_length_range(graph, node_range, cutoff):
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
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    print("Calculating all pairs shorttest path in parallel.")
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
    print("Done calculating all pairs shorttest path in parallel.")
    return dists_dict


def precompute_dist_data(graph, num_nodes, approximate=4):
    """
    Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
    :return:
    """
    n = num_nodes
    dists_array = np.zeros((n, n))
    dists_array2 = np.zeros((n, n))
    # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
    # dists_dict = {c[0]: c[1] for c in dists_dict}
    dists_dict = all_pairs_shortest_path_length_parallel(graph, cutoff=approximate if approximate > 0 else None)

    return dists_dict


# %%
if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    udbms_dataset = UDBMSDataset()
    input_file_name = os.path.join(dirname, './generated/person_duplicate_without_subject_feat_25percent.graph')
    attributes_file_name = os.path.join(dirname, './data/attributes.txt')
    prefix = './data/duplicate_without_subject_feat_25percent_missing_edges_'
    udbms_dataset.main(prefix, input_file_name, attributes_file_name)
