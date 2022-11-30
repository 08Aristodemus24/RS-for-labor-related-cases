import json
import pickle
from random import shuffle

f1_name = "./data/clean_more_than_subject_person_subjectdup-train.dat"
f2_name = "./data/clean_more_than_subject_person_subjectdup-val.dat"
f3_name = "./data/clean_more_than_subject_person_subjectdup-test.dat"
f4_name = "./data/clean_more_than_subject_person_subjectdup-neg-train.dat"
f5_name = "./data/clean_more_than_subject_person_subjectdup-neg-val.dat"
f6_name = "./data/clean_more_than_subject_person_subjectdup-neg-test.dat"

featmap_name = "./data/clean_more_than_subject_person_subjectdup-features_map_legend.json"
grouped_name = "./data/clean_more_than_subject_person_subjectdup-grouped.json"

# loading data
with open(f1_name, 'rb') as f1, open(f2_name, 'rb') as f2, open(f3_name, 'rb') as f3, \
        open(f4_name, 'rb') as f4, open(f5_name, 'rb') as f5, open(f6_name, 'rb') as f6:
    duplicate_train_list = pickle.load(f1)
    duplicate_val_list = pickle.load(f2)
    duplicate_test_list = pickle.load(f3)

    duplicate_train_neg_list = pickle.load(f4)
    duplicate_val_neg_list = pickle.load(f5)
    duplicate_test_neg_list = pickle.load(f6)

attr_feat_map = json.load(open(featmap_name))
grouped_persons_map = json.load(open(grouped_name))

# generating header line
header_line = "id,label,"
separator = ","

header_attributes = []
attribute_map = {}

for attribute in attr_feat_map.keys():
    header_attributes.append("left_" + attribute)
    header_attributes.append("right_" + attribute)
    attribute_map[attribute] = "-"

header_line += separator.join(header_attributes)

# generate map for each person and its attributes
skip_attribute_list = ['relation', 'relative', 'spouse', 'child', 'parent', 'partner', 'predecessor', 'successor',
                       'opponent', 'rival', 'subject']

persons_attributes_map = {}

for person_name, person_value in grouped_persons_map.items():
    attr_person = grouped_persons_map.get(person_name)
    attr_map_person = attribute_map.copy()

    for tup in attr_person:
        if tup[0] not in skip_attribute_list:
            attr_map_person[tup[0]] = tup[1]

    persons_attributes_map[person_name] = attr_map_person


# converting list to csv lines [function]
def get_csv_data_lines(duplicate_list, sample_label="1", id_offset=0):
    id_count = 0 + id_offset
    csv_lines = []

    for duplicate in duplicate_list:
        first = duplicate[0]
        second = duplicate[1]

        for i in range(len(first)):
            person1 = first[i]
            person2 = second[i]
            person_line = [str(id_count), sample_label]  # sample_label => 0: no dup / 1: dup
            id_count += 1

            attr_person1 = persons_attributes_map.get(person1)
            attr_person2 = persons_attributes_map.get(person2)

            for attr in attr_feat_map.keys():
                person_line.append(attr_person1[attr])
                person_line.append(attr_person2[attr])

            csv_lines.append(separator.join(person_line))

    return csv_lines, id_count


# create csv data out of sample sets using function
pos_train_set, offset = get_csv_data_lines(duplicate_train_list, "1")
neg_train_set, _ = get_csv_data_lines(duplicate_train_neg_list, "0", id_offset=offset)
train_set = [*pos_train_set, *neg_train_set]

pos_test_set, offset = get_csv_data_lines(duplicate_test_list, "1")
neg_test_set, _ = get_csv_data_lines(duplicate_test_neg_list, "0", id_offset=offset)
test_set = [*pos_test_set, *neg_test_set]

pos_val_set, offset = get_csv_data_lines(duplicate_val_list, "1")
neg_val_set, _ = get_csv_data_lines(duplicate_val_neg_list, "0", id_offset=offset)
val_set = [*pos_val_set, *neg_val_set]

# shuffle(train_set)
# shuffle(test_set)
# shuffle(val_set)

train_csv_lines = [header_line, *train_set]
test_csv_lines = [header_line, *test_set]
val_csv_lines = [header_line, *val_set]

with open('./sets/train.csv', 'w') as file:
    for line in train_csv_lines:
        file.write(line + "\r\n")

with open('./sets/test.csv', 'w') as file:
    for line in test_csv_lines:
        file.write(line + "\r\n")

with open('./sets/val.csv', 'w') as file:
    for line in val_csv_lines:
        file.write(line + "\r\n")
