import math


def calculate_dataset_entropy(feature_label):
    frequencies = [0] * 4
    num_entries = len(feature_label)
    for i in range(num_entries):
        frequencies[feature_label[1]] += 1
    probabilities = [frequency / num_entries for frequency in frequencies]
    entropy = 0
    for probability in probabilities:
        entropy -= probability * math.log2(probability)
    return entropy


def find_best_split(dataset, column_index):
    split_value = None
    max_info_gain = -1
    return split_value, max_info_gain


def find_split(dataset):
    """find_split

    :param data: matrix of input data, where each column except the last
    represents a feature and the last column represents the labels. """

    split_column_index = 0
    best_split_value = 0
    max_info_gain = -1

    for i in range(len(dataset[0]) - 1):
        split_value, info_gain = find_best_split(dataset, i)
        if info_gain > max_info_gain:
            split_column_index = i
            best_split_value = split_value
            max_info_gain = info_gain

    matrix_le = dataset[dataset[split_column_index] <= best_split_value]
    matrix_g = dataset[dataset[split_column_index] > best_split_value]

    return lambda data: data[split_column_index] <= best_split_value, \
           matrix_le, matrix_g
