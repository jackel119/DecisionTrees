import math
import numpy

NUM_ROOMS = 4


def calculate_dataset_entropy(labels):
    frequencies = [0] * NUM_ROOMS
    for label in labels:
        frequencies[label - 1] += 1
    probabilities = [frequency / len(labels) for frequency in frequencies]
    entropy = 0
    for probability in probabilities:
        entropy -= probability * math.log2(probability)
    return entropy


def find_best_split(dataset, column_index):
    split_value = None
    min_remainder = math.inf
    feature_label = dataset[:, [column_index, -1]]
    sorted_feature_label = numpy.sort(feature_label, axis=0)
    # TODO: Find splits, calculate remainder, find minimum remainder
    return split_value, min_remainder


def find_split(dataset):
    """find_split

    :param data: matrix of input data, where each column except the last
    represents a feature and the last column represents the labels. """

    split_column_index = 0
    best_split_value = 0
    max_info_gain = -1
    label_column = [row[-1] for row in dataset]
    h_dataset = calculate_dataset_entropy(label_column)

    for i in range(dataset.shape[1] - 1):
        split_value, remainder = find_best_split(dataset, i)
        info_gain = h_dataset - remainder

        if info_gain > max_info_gain:
            split_column_index = i
            best_split_value = split_value
            max_info_gain = info_gain

    matrix_le = dataset[dataset[split_column_index] <= best_split_value]
    matrix_g = dataset[dataset[split_column_index] > best_split_value]

    return lambda data: data[split_column_index] <= best_split_value, \
           matrix_le, matrix_g
