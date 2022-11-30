import pickle
import networkx as nx
import multiprocessing as mp
import numpy as np

import random
import requests
import json
import asyncio
import re
import os.path

from random import shuffle
from datetime import datetime, date
from itertools import islice, combinations
from collections import defaultdict


# %%
class RawDataset:
    G = nx.Graph()
    id_map = {}
    geo_data = {}
    linked_nodes_map = {}
    nodes_list = []
    links_list = []

    data = None
    prepared_data = None
    duplicate_mapping = []
    attributes_map = {}
    attributes_embedding_type_map = {}
    attributes_raw_features = {}

    rel_list = ['relation', 'relative', 'spouse', 'child', 'parent', 'partner', 'predecessor', 'successor',
                'opponent', 'rival']

    embedding_types = {
        -1: "none",
        0: "categorical type",
        1: "date (3dim vec)",
        2: "number",
        3: "geolocation (2dim vec)",
        4: "word2vec"
    }

    fixed_start_date = date(2019, 9, 20)

    # main data load function
    def load_data(self, dataset_name, select_random=False):
        """
        :param dataset_name:
        """
        dataset_dir = './generated/' + dataset_name
        print("Loading data...")

        with open(dataset_dir + ".graph", "r") as f:
            self.data = f.readlines()

            if select_random is True:
                shuffle(self.data)

        print("Done.")

    # attributes load function
    def load_attributes(self, attribute_file_path):
        """
        :param dataset_name:
        """
        dataset_dir = './generated/' + attribute_file_path
        print("Loading attributes...")

        with open(dataset_dir + ".txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                tup = tuple(line.replace("\n", "").split(","))
                self.attributes_map[tup[0]] = {}
                self.attributes_embedding_type_map[tup[0]] = int(tup[1])

        print("Done.")

    # geo data load function
    def load_geo_data(self, geo_data_file):
        """
        :param dataset_name:
        """
        file_path = './generated/' + geo_data_file + ".json"
        print("Loading geo data...")

        self.geo_data = json.load(open(file_path))

        print("Done.")

    def get_next_n_data_points(self, count=-1, ignore_subjects=True):
        """
        :param ignore_subjects:
        :param count:
        :return: next_n_tuples
        """
        if count == -1:
            count = len(self.data)

        tuples = []

        print("Retrieving " + str(count) + " tuples...")
        for line in islice(self.data, count):
            tup = tuple(line.replace("\n", "").split("\t"))
            if len(tup) != 3:
                print("Each line must be a triple. Line was: ", line)
                continue

            # ignoring subject entries
            if ignore_subjects is True and tup[1] == 'subject':
                continue

            tuples.append(tup)

            print_progress_bar(len(tuples), count, prefix="tuples: ", length=50)

        print("Done.")
        print("")

        return tuples

    def group_tuples_by_unique_name(self, count=-1, ignore_subjects=True):
        """
        :param ignore_subjects:
        :param count:
        :return: grouped_tuples
        """
        if count == -1:
            count = len(self.data)

        items = self.get_next_n_data_points(count, ignore_subjects)
        print("Grouping tuples...")

        group = defaultdict(list)
        for k, *v in items:
            group[k].append(v)

        print("")
        print("detected groups (persons): " + str(len(group)))
        print("======================================")
        print("")

        self.prepared_data = {**dict(group)}

    def group_tuples_by_subject(self, included_nodes=None, count=-1):
        """
        :param included_nodes: set of nodes which should be included only, if None all nodes
        :param count:
        :return: grouped_tuples
        """
        if count == -1:
            count = len(self.data)

        items = self.get_next_n_data_points(count, ignore_subjects=False)
        print("Grouping tuples by subject...")

        group = defaultdict(list)
        for k, *v in items:
            if v[0] == 'subject':
                if k in included_nodes or included_nodes is None:
                    group[v[1]].append(k)

        print("")
        print("detected subjects: " + str(len(group)))
        print("======================================")
        print("")

        return group

    def start_duplication(self, percentage=0.2, manipulate_attr=False):
        """
        :param manipulate_attr:
        :param tuple_group:
        :param percentage:
        """

        unique_tuples = self.choose_unique_tuples_for_duplication(self.prepared_data, percentage)
        duplicate_tuples = []
        duplicate_mapping = []

        for duplicate_tup in unique_tuples:
            conv_tuple = list(duplicate_tup)

            if manipulate_attr is True:
                new_attributes = []
                for attributes in conv_tuple[1]:
                    raise NotImplementedError

                conv_tuple[1] = new_attributes

            duplicate_name = conv_tuple[0] + "_(duplicate)"

            duplicate_mapping.append((conv_tuple[0], duplicate_name))
            conv_tuple[0] = duplicate_name
            duplicate_tuples.append(tuple(conv_tuple))

        self.prepared_data = {**dict(duplicate_tuples), **dict(self.prepared_data)}
        self.duplicate_mapping = duplicate_mapping

        print("")
        print("new duplicate mappings: " + str(len(self.duplicate_mapping)))
        print("=> groups/persons (now): " + str(len(self.prepared_data)))
        print("======================================")
        print("")

    def save_duplicate_data_graph(self, dataset_name, check_features=False):
        dataset_dir = './generated/' + dataset_name
        print("Saving graph to directory...")

        with open(dataset_dir + ".graph", "w") as f:
            for key, value in self.prepared_data.items():
                for attribute in value:
                    skip = False
                    if check_features is True and attribute[0] in self.rel_list:
                        if attribute[1] not in self.prepared_data.keys():
                            skip = True

                    # added skip boolean for removing nodes with no features
                    if not skip:
                        f.write(key + '\t' + attribute[0] + '\t' + attribute[1] + '\n')

    def get_attribute_dispersion_in_all_nodes(self, print_result=False):
        attributes_map = self.attributes_map

        for attribute in attributes_map:
            attributes_map.update({attribute: 0})

        for key, value in self.prepared_data.items():
            already_used_attributes = []
            for attributes in value:
                if attributes[0] not in already_used_attributes:
                    attributes_map[attributes[0]] += 1
                    already_used_attributes.append(attributes[0])

        attributes_map = sorted(attributes_map.items(), key=lambda x: x[1], reverse=True)
        node_num = len(self.prepared_data)

        if print_result is True:
            for tup in attributes_map:
                print(tup[0] + ": ", tup[1], "(" + str(tup[1] / node_num * 100) + "%)")

        return attributes_map

    def create_feature_map_legend(self):
        # creating feature map legend (dict with attributes, each is containing a dict with all attributes values and an ascending number
        print("Creating feature map legend...")

        for key, value in self.prepared_data.items():
            for attributes in value:
                if attributes[0] in ["activeYearsEndYear", "activeYearsStartYear", "birthYear", "deathYear",
                                     "birthDate", "deathDate", "height", "networth", "salary", "weight", "title",
                                     "alias", "birthName"]:
                    if '"' in attributes[1]:
                        attributes[1] = attributes[1].split('"')[1]

                attribute_name = attributes[0]
                attribute_value = attributes[1]

                specific_attribute_map = self.attributes_map.get(attribute_name)

                if specific_attribute_map.get(attribute_value) is None:
                    specific_attribute_map.update({attribute_value: len(specific_attribute_map)})
                    self.attributes_map.update({attribute_name: specific_attribute_map})

        # with open("./generated/feature.dat", 'w') as f:
        #     json.dump(self.attributes_map, f)

        print("=> created feature map legend for " + str(len(self.attributes_map)) + " attributes")
        print("======================================")
        print("")

    def get_geo_locations_for_attribute_values(self, geo_cache_name, use_cache=True, check_zeros=False):
        print("Calling Geolocation API...")

        geo_locations_map = {}
        file_path = "./generated/" + geo_cache_name + ".geo.json"

        if use_cache is True and os.path.isfile(file_path):
            geo_locations_map = json.load(open(file_path))

        places = []
        for key, value in self.prepared_data.items():
            for attributes in value:
                if attributes[0] in ["birthPlace", "deathPlace", "hometown", "residence", "restingPlace",
                                     "stateOfOrigin", "almaMater"]:
                    look_up_place = attributes[1].replace("_", " ")
                    look_up_place = re.sub(r"\(.*\d+\)", "", look_up_place)

                    if geo_locations_map.get(attributes[1]) is None:
                        places.append((attributes[0], attributes[1], look_up_place))

        if check_zeros is True:
            for key, value in geo_locations_map.items():
                if value.get("lat") == 0 and value.get("lng") == 0:
                    look_up_place = key.replace("_", " ")
                    look_up_place = re.sub(r"\(.*\d+\)", "", look_up_place)
                    places.append(("zero", key, look_up_place))

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.api_request(places))

        for i in range(len(results)):
            data = results[i]

            try:
                geo_locations_map[places[i][1]] = data.get("results")[0].get("geometry").get("location")
                print(places[i][1], "#", places[i][0], "lat:",
                      data.get("results")[0].get("geometry").get("location").get("lat"), "lon:",
                      data.get("results")[0].get("geometry").get("location").get("lng"))
            except:
                geo_locations_map[places[i][1]] = {"lat": 0, "lng": 0}
                print("error by fetching:", places[i][1], " | using query entity: ", places[i][2])

        with open("./generated/" + geo_cache_name + ".geo.json", 'w') as f:
            json.dump(geo_locations_map, f)

    def create_features(self, use_subject=False):
        print("Creating features...")

        np.seterr(all='raise')

        for attribute_name, attribute_values in self.attributes_map.items():
            method_key = self.attributes_embedding_type_map.get(attribute_name)

            if method_key == 0:
                attribute_raw_features = np.zeros([len(self.id_map), len(attribute_values)], dtype=np.int)
            elif method_key == 3:
                attribute_raw_features = np.zeros([len(self.id_map), 2], dtype=np.float)
            else:
                attribute_raw_features = np.zeros([len(self.id_map), 1], dtype=np.float)
            self.attributes_raw_features[attribute_name] = attribute_raw_features

        for key, value in self.id_map.items():
            person = self.prepared_data.get(key)

            if person is None:
                continue

            for attribute_name, attribute_values in self.attributes_map.items():
                attribute_raw_features = self.attributes_raw_features[attribute_name]
                method_key = self.attributes_embedding_type_map.get(attribute_name)

                for tup in person:
                    if tup[0] == attribute_name:
                        if method_key == 0:
                            attr_id = attribute_values.get(tup[1])
                            attribute_raw_features[value - 1, attr_id] = 1

                        elif method_key == 1:
                            date = datetime.strptime(tup[1], '%Y-%m-%d').date()
                            days_difference = int((self.fixed_start_date - date).days)

                            attribute_raw_features[value - 1, 0] = abs(days_difference)

                        elif method_key == 2:
                            attribute_raw_features[value - 1, 0] = abs(float(tup[1]))

                        elif method_key == 3:
                            geolocation = self.geo_data.get(tup[1])
                            attribute_raw_features[value - 1, 0] = geolocation.get("lat")
                            attribute_raw_features[value - 1, 1] = geolocation.get("lng")

                        elif method_key == 4 and use_subject is True:
                            attribute_raw_features[value - 1, 0] = tup[1]

                self.attributes_raw_features[attribute_name] = attribute_raw_features

            print_progress_bar(value + 1, len(self.id_map), prefix="persons: ",
                               suffix=str(value + 1) + "/" + str(len(self.id_map)), length=50)

        for attribute_name, attribute_values in self.attributes_map.items():
            attribute_raw_features = self.attributes_raw_features[attribute_name]
            method_key = self.attributes_embedding_type_map.get(attribute_name)

            # do normalization on attribute features with large numbers, e.g. day difference and raw numbers
            if method_key == 1 or method_key == 2:
                attribute_raw_features[:, 0] = np.log(attribute_raw_features[:, 0] + 1.0)
                attribute_raw_features[:, 0] = np.log(
                    attribute_raw_features[:, 0] - min(np.min(attribute_raw_features[:, 0]), -1))

            self.attributes_raw_features[attribute_name] = attribute_raw_features

        print("Done.")

    def create_nodes(self):
        id_count = 0
        print("Adding nodes to graph...")

        for key, value in self.prepared_data.items():
            if dataset.id_map.get(key) is None:
                self.id_map[key] = id_count
                self.G.add_node(key)

                node_map = {"test": False, "val": False, "id": key, "features": []}
                self.nodes_list.append(node_map)

                id_count += 1

            for attributes in value:
                if attributes[0] in self.rel_list:
                    if self.id_map.get(attributes[1]) is None:
                        self.id_map[attributes[1]] = id_count
                        self.G.add_node(attributes[1])

                        node_map = {"test": False, "val": False, "id": attributes[1], "features": []}
                        self.nodes_list.append(node_map)

                        id_count += 1

        print("=> added nodes: " + str(len(self.G.nodes)))
        print("======================================")
        print("")

    def create_edges(self, input_dataset_name="", check_missing_edges=False, use_cache=True):
        print("Adding edges to graph...")

        for key, value in self.prepared_data.items():
            for attributes in value:
                if attributes[0] in self.rel_list:
                    linked_nodes = self.make_label(key, attributes[1])
                    if self.linked_nodes_map.get(linked_nodes) is None:
                        link_map = {"source": key, "target": attributes[1]}

                        self.links_list.append(link_map)
                        self.G.add_edge(key, attributes[1])
                        self.linked_nodes_map[linked_nodes] = attributes[0]

        print("=> added edges: " + str(len(self.G.edges)))
        print("======================================")
        print("")

        if check_missing_edges is True:
            print("Checking for missing edges in graph caused by duplication...")

            # loading duplicate mappings
            with open("./generated/" + input_dataset_name + ".map", 'rb') as file:
                duplicate_node_mappings = pickle.load(file)

            if use_cache is True and os.path.isfile("./generated/" + input_dataset_name + ".dist"):
                with open("./generated/" + input_dataset_name + ".dist", 'rb') as file:
                    dist = pickle.load(file)
            else:
                dist = precompute_dist_data(self.G, 0, approximate=4)
                with open("./generated/" + input_dataset_name + ".dist", 'wb') as file:
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
                            # print("edge missing between ", duplicate, " and ", neighbor)
                            count += 1
                            linked_nodes = self.make_label(duplicate, neighbor)
                            if self.linked_nodes_map.get(linked_nodes) is None:
                                link_map = {"source": duplicate, "target": neighbor}
                                self.links_list.append(link_map)
                                self.G.add_edge(duplicate, neighbor)

            print("=> added missing edges: " + str(count))
            print("======================================")
            print("")

    def generate_duplicates_from_subject_ground_truth(self, entry_offset=0, path_threshold=2,
                                                      only_connected_components=True,
                                                      print_examples=True, example_count=5):
        """
        Function generates the duplicate mapping based on similiar subject groups.
        :param example_count:
        :param print_examples:
        :param only_connected_components: whether only connected nodes in the graph should be included
        :param entry_offset: how many entries per subject should be included (starting at 0 => min. 2 entries/subj.)
        :param path_threshold: maximum length of path between pair nodes
        """

        print("Calculating subject ground truth...")

        if only_connected_components is True:
            included_nodes = set()
            comps = [comp for comp in nx.connected_components(self.G) if len(comp) > 10]

            for comp in comps:
                for node in comp:
                    included_nodes.add(node)

            grouped_tuples_sub = self.group_tuples_by_subject(included_nodes=included_nodes)
        else:
            grouped_tuples_sub = self.group_tuples_by_subject()

        example = 0
        pair_count = 0
        involved_nodes = set()
        unique_pairs = set()
        duplicate_mapping = []

        for key, value in grouped_tuples_sub.items():
            length = len(value)

            if 1 < length <= entry_offset + 2:
                pairs_to_compare = list(combinations(value, 2))

                for pair in pairs_to_compare:
                    if self.make_label(pair[0], pair[1]) in unique_pairs:
                        continue

                    unique_pairs.add(self.make_label(pair[0], pair[1]))

                    if nx.has_path(self.G, pair[0], pair[1]):
                        if nx.shortest_path_length(self.G, pair[0], pair[1]) >= path_threshold:
                            if print_examples is True and example < example_count:
                                print("")
                                print("---------------- Example", example + 1, "---")
                                print("path length:", nx.shortest_path_length(dataset.G, pair[0], pair[1]))
                                print("")
                                print("subject:", key)
                                print("pair:", pair[0], " || ", pair[1])
                                print("")

                                print("attribute 0:", dataset.prepared_data.get(pair[0]))
                                print("")
                                print("attribute 1:", dataset.prepared_data.get(pair[1]))
                                print("----------------")
                                print("")
                                example += 1

                            pair_count += 1
                            involved_nodes.add(value[0])
                            involved_nodes.add(value[1])
                            duplicate_mapping.append((value[0], value[1]))

        print("=> involved nodes: ", len(involved_nodes))
        print("=> duplicate pairs count: ", pair_count)
        print("======================================")
        print("")

        self.duplicate_mapping = duplicate_mapping

    def save_all(self, prefix, save_grouped_data=False, save_duplicates=False):
        self.save_id_map(prefix)
        self.save_features_map_legend(prefix)
        self.save_features(prefix)
        self.save_graph(prefix)

        if save_duplicates is True:
            self.save_duplicate_mapping(prefix)

        if save_grouped_data is True:
            self.save_grouped_data(prefix)

    def save_duplicate_mapping(self, prefix):
        print("Saving duplicate mapping to directory...")
        print("")

        filename = "./generated/" + prefix + ".map"
        with open(filename, 'wb') as fp:
            pickle.dump(self.duplicate_mapping, fp)

    def save_id_map(self, prefix):
        print("")
        print("Saving graph id map...")

        filename = "./data/" + prefix + "-id_map.json"
        with open(filename, 'w') as f:
            json.dump(self.id_map, f)

    def save_features_map_legend(self, prefix):
        print("")
        print("Saving features map legend...")

        json_data = self.attributes_map
        features_map_legend_file = "./data/" + prefix + "-features_map_legend.json"
        with open(features_map_legend_file, 'w') as f:
            json.dump(json_data, f)

    def save_grouped_data(self, prefix):
        print("")
        print("Saving grouped data...")

        json_data = self.prepared_data
        grouped_data_file = "./data/" + prefix + "-grouped.json"
        with open(grouped_data_file, 'w') as f:
            json.dump(json_data, f)

    def save_features(self, prefix):
        print("")
        print("Saving features...")

        features_file = "./data/" + prefix + "-feats.dat"
        with open(features_file, 'wb') as file:
            pickle.dump(self.attributes_raw_features, file, protocol=4)

    def save_graph(self, prefix):
        print("")
        print("Saving graph...")
        print("Number of edges: ", self.G.number_of_edges())
        print("Number of nodes: ", self.G.number_of_nodes())

        filename = "./data/" + prefix + "-G.json"
        output_map = {"directed": False}
        meta_data_map = {"name": "udbms"}

        output_map["graph"] = meta_data_map
        output_map["nodes"] = self.nodes_list
        output_map["links"] = self.links_list
        output_map["multigraph"] = False

        with open(filename, 'w') as f:
            json.dump(output_map, f)

        print("Done.")

    @staticmethod
    async def api_request(places):
        api_endpoint = "https://maps.googleapis.com/maps/api/geocode/json?&address="

        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                None,
                requests.get,
                api_endpoint + place[2] + "&key=AIzaSyBe-bHNwQ4ksDyxLCTb4jrRWDm4NF_KnWI"
            )
            for place in places  # islice(places, 0, 2500) # for fetching only 2500 per time
        ]

        results = []
        for response in await asyncio.gather(*futures):
            data = response.json()
            results.append(data)

        return results

    @staticmethod
    def choose_unique_tuples_for_duplication(tuple_group, percentage=0.5):
        """
        :param tuple_group:
        :param percentage:
        :return: unique_tuples
        """
        count = int(percentage * len(tuple_group))

        print("Selecting " + str(count) + " tuples for duplication... (" + str(percentage * 100) + "%)")
        n_items = list(islice(tuple_group.items(), count))

        return n_items

    @staticmethod
    def make_label(node1, node2):
        if node1 < node2:
            return node1 + "_" + node2
        return node2 + "_" + node1


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

    # print("Calculating all pairs shorttest path in parallel.")
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
    # print("Done calculating all pairs shorttest path in parallel.")
    return dists_dict


def precompute_dist_data(graph, num_nodes, approximate=4):
    return all_pairs_shortest_path_length_parallel(graph, cutoff=approximate if approximate > 0 else None)


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


# %%
if __name__ == "__main__":
    # start duplication on clean dataset
    dataset = RawDataset()
    dataset.load_data('clean_more_than_subject_person', select_random=True)
    dataset.load_attributes('person.attributes')
    dataset.load_geo_data('person.geo')

    dataset.group_tuples_by_unique_name(ignore_subjects=False)

    # dataset.start_duplication(percentage=0, manipulate_attr=False)
    # dataset.get_attribute_dispersion_in_all_nodes(grouped_tuples, print_result=True)
    # dataset.get_geo_locations_for_attribute_values("person", use_cache=True, check_zeros=True)

    duplicate_dataset_name = 'clean_more_than_subject_person_subjectdup'
    dataset.save_duplicate_data_graph(duplicate_dataset_name, check_features=False)

    dataset.create_feature_map_legend()

    dataset.create_nodes()
    dataset.create_edges(check_missing_edges=False, use_cache=False, input_dataset_name=duplicate_dataset_name)

    dataset.generate_duplicates_from_subject_ground_truth(entry_offset=2, path_threshold=2,
                                                          only_connected_components=True,
                                                          print_examples=False, example_count=3)

    dataset.create_features(use_subject=False)

    dataset.save_all(duplicate_dataset_name, save_grouped_data=True, save_duplicates=True)
