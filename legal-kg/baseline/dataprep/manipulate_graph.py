import sys
import json
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from itertools import islice
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


# %%
class Dataset:
    G = nx.Graph()
    graphs = None
    features = None
    orig_features = None
    edge_labels = None
    dataset_str = ""

    # main data load function
    def load_graphs(self, dataset_str):
        dataset_dir = './data/' + dataset_str
        print("Loading data...")

        self.dataset_str = dataset_str
        self.G = json_graph.node_link_graph(json.load(open(dataset_dir + "-G.json")))
        edge_labels_internal = json.load(open(dataset_dir + "-class_map.json"))
        edge_labels_internal = {i: l for i, l in edge_labels_internal.items()}

        train_ids = [n for n in self.G.nodes()]
        train_labels = np.array([edge_labels_internal[i] for i in train_ids])
        if train_labels.ndim == 1:
            train_labels = np.expand_dims(train_labels, 1)

        self.edge_labels = train_labels

        print("Using only features..")
        feats = np.load(dataset_dir + "-feats.npy")
        orig_feats = feats

        feats[:, 0] = np.log(feats[:, 0] + 1.0)
        feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
        feat_id_map = json.load(open(dataset_dir + "-id_map.json"))
        feat_id_map = {id: val for id, val in feat_id_map.items()}
        train_feats = feats[[feat_id_map[id] for id in train_ids]]
        orig_train_feats = orig_feats[[feat_id_map[id] for id in train_ids]]

        node_dict = {}
        for id, node in enumerate(self.G.nodes()):
            node_dict[node] = id

        print("Finding connected components..")
        comps = [comp for comp in nx.connected_components(self.G) if len(comp) > 10]
        print("Finding subgraphs..")
        self.graphs = [self.G.subgraph(comp) for comp in comps]
        print("Number of Graphs in input:")
        print(len(self.graphs))

        id_all = []
        for comp in comps:
            id_temp = []
            for node in comp:
                id = node_dict[node]
                id_temp.append(id)
            id_all.append(np.array(id_temp))

        print("Creating features")
        self.features = [train_feats[id_temp, :] + 0.1 for id_temp in id_all]
        self.orig_features = [orig_train_feats[id_temp, :] for id_temp in id_all]

    # select sub graph method
    def get_sub_graph(self, number):
        return SubGraph(self.graphs[number], number, self.dataset_str)


# %%
class SubGraph:
    old_G = nx.Graph()
    new_G = None
    number = None
    duplicates = []
    dataset_str = ""

    def __init__(self, graph, number, dataset_str):
        self.old_G = graph
        self.new_G = nx.Graph(graph)
        self.number = number
        self.dataset_str = dataset_str

    def duplicate_nodes(self, count=2):
        for new_node in islice(self.old_G, count):
            features = nx.get_node_attributes(self.new_G, 'features')
            test = nx.get_node_attributes(self.new_G, 'test')
            val = nx.get_node_attributes(self.new_G, 'val')

            self.new_G.add_node(new_node + "_duplicate", features=features[new_node], test=test[new_node],
                                val=val[new_node])

            for neighbor_node in list(nx.all_neighbors(self.new_G, new_node)):
                self.new_G.add_edge(new_node + "_(duplicate)", neighbor_node)

            self.duplicates.append((new_node, new_node + "_(duplicate)"))

    def draw_both_graphs(self):
        fig, ax = plt.subplots(2)

        color_map = []
        for node in self.new_G:
            if node.endswith('_(duplicate)'):
                color_map.append('red')
            else:
                color_map.append('blue')

        nx.draw(self.old_G, with_labels=True, node_color='b', font_size=5, ax=ax[0],
                pos=nx.kamada_kawai_layout(self.old_G))

        nx.draw(self.new_G, with_labels=True, node_color=color_map, font_size=5, ax=ax[1],
                pos=nx.kamada_kawai_layout(self.new_G))

        plt.savefig("./plots/" + self.dataset_str + "-subgraph_" + str(self.number) + ".png", bbox_inches="tight", dpi=600)
        plt.show()

    def draw_graph(self):

        color_map = []
        for node in self.new_G:
            if node.endswith('_(duplicate)'):
                color_map.append('red')
            else:
                color_map.append('blue')

        nx.draw(self.new_G, with_labels=True, node_color=color_map, font_size=5,
                pos=nx.kamada_kawai_layout(self.new_G))

        plt.savefig("./plots/" + self.dataset_str + "-subgraph_" + str(self.number) + ".png", bbox_inches="tight", dpi=600)
        plt.show()


# %%
if __name__ == "__main__":
    dataset = Dataset()

    if sys.argv[1] == "udbms":
        dataset.load_graphs('udbms')
        sub_graph = dataset.get_sub_graph(int(sys.argv[2]))
        sub_graph.duplicate_nodes(3)
        sub_graph.draw_both_graphs()

    elif sys.argv[1] in ["d_udbms", "duplicate_udbms"]:
        dataset.load_graphs('duplicate_udbms')
        sub_graph = dataset.get_sub_graph(int(sys.argv[2]))
        sub_graph.draw_graph()
