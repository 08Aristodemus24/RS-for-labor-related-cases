import pickle
from itertools import islice
from collections import defaultdict
from random import shuffle


# %%
class RawDataset:
    data = None
    duplicate_data = None
    duplicate_mapping = []

    # main data load function
    def load_data(self, dataset_name):
        dataset_dir = './generated/' + dataset_name
        print("Loading data...")

        with open(dataset_dir + ".graph", "r") as f:
            self.data = f.readlines()
            # shuffle(self.data)

        print("Done.")

    def get_next_n_data_points(self, count=-1):
        """
        :param count:
        :return: next_n_tuples
        """
        if count == -1:
            count = len(self.data)

        tuples = []
        relation_entries = []

        print("Retrieving " + str(count) + " tuples...")
        for line in islice(self.data, count):
            tup = tuple(line.replace("\n", "").split("\t"))
            if len(tup) != 3:
                print("Each line must be a triple. Line was: ", line)
                continue

            # ignoring subject entries
            if tup[1] == 'subject':
                continue

            tuples.append(tup)

        return tuples

    def group_tuples_by_unique_name(self, count=-1):
        """
        :param count:
        :return: grouped_tuples
        """
        if count == -1:
            count = len(self.data)

        items = self.get_next_n_data_points(count)
        print("Grouping tuples...")
        group = defaultdict(list)
        for k, *v in items:
            group[k].append(v)

        print("")
        print("detected groups (persons): " + str(len(group)))
        print("======================================")
        print("")

        return group

    def start_duplication(self, tuple_group, percentage=0.2, manipulate_attr=False):
        """
        :param manipulate_attr:
        :param tuple_group:
        :param percentage:
        """

        unique_tuples = self.choose_unique_tuples_for_duplication(tuple_group, percentage)
        duplicate_tuples = []
        duplicate_mapping = []

        for duplicate_tup in unique_tuples:
            conv_tuple = list(duplicate_tup)

            if manipulate_attr is True:
                new_attributes = []
                for attributes in conv_tuple[1]:
                    # d = ["nothing 0.35", "1 0.2", "2 0.1", "3 0.05", "1 0.15", "2 0.1", "3 0.05"]
                    if attributes[0] in ["birthDate", "deathDate"]:
                        value = attributes[1].split('"')[1]
                        # new_attributes.append([attributes[0], manipulate_date(value)])
                    elif attributes[0] in ["activeYearsEndYear", "activeYearsStartYear", "birthYear", "deathYear"]:
                        value = int(attributes[1].split('"')[1])
                        # new_attributes.append([attributes[0], manipulate_year(value)])
                    elif attributes[0] in ["height", "networth", "salary", "weight"]:
                        value = float(attributes[1].split('"')[1])
                        # new_attributes.append([attributes[0], manipulate_number(value)])
                    else:
                        value = attributes[1].replace('_', ' ')
                        # new_attributes.append([attributes[0], manipulate_data(value)])

                conv_tuple[1] = new_attributes

            duplicate_name = conv_tuple[0] + "_(duplicate)"

            duplicate_mapping.append((conv_tuple[0], duplicate_name))
            conv_tuple[0] = duplicate_name
            duplicate_tuples.append(tuple(conv_tuple))

        self.duplicate_data = {**dict(duplicate_tuples), **dict(tuple_group)}
        self.duplicate_mapping = duplicate_mapping

        print("")
        print("new duplicate mappings: " + str(len(self.duplicate_mapping)))
        print("=> groups/persons (now): " + str(len(self.duplicate_data)))
        print("======================================")
        print("")
        exit(0)

    def save_duplicate_data_graph(self, dataset_name, check_features=False):
        dataset_dir = './generated/' + dataset_name
        print("Saving graph to directory...")

        rel_list = ['relation', 'relative', 'spouse', 'child', 'parent', 'partner', 'predecessor',
                    'successor', 'opponent', 'rival']

        with open(dataset_dir + ".graph", "w") as f:
            for key, value in self.duplicate_data.items():
                for attribute in value:
                    skip = False
                    if check_features is True and attribute[0] in rel_list:
                        if attribute[1] not in self.duplicate_data.keys():
                            skip = True

                    # added skip boolean for removing nodes with no features
                    if not skip:
                        f.write(key + '\t' + attribute[0] + '\t' + attribute[1] + '\n')

        print("Saving duplicate mapping to directory...")

        with open(dataset_dir + ".map", 'wb') as fp:
            pickle.dump(self.duplicate_mapping, fp)

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


# %%
if __name__ == "__main__":
    # # clean raw dataset and remove features
    # dataset = RawDataset()
    # dataset.load_data('person')
    # grouped_tuples = dataset.group_tuples_by_unique_name()
    # dataset.start_duplication(grouped_tuples, 0)
    # dataset.save_duplicate_data_graph('person_without_subject_feat', check_features=True)

    # start duplication on clean dataset
    dataset = RawDataset()
    dataset.load_data('person_without_subject_feat')
    grouped_tuples = dataset.group_tuples_by_unique_name()
    dataset.start_duplication(grouped_tuples, percentage=0.25, manipulate_attr=True)
    dataset.save_duplicate_data_graph('person_duplicate_without_subject_feat_25percent_attr', check_features=False)
