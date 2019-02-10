import math

import numpy as np


def calculate_dataset_entropy(dataset):
    """calculate the entropy of the dataset

    :param dataset: 2d np array
    """
    # compute labels
    labels = [row[-1] for row in dataset]
    # compute frequency of each label
    _, frequencies = np.unique(labels, return_counts=True)
    # compute each label's probablity
    probabilities = [frequency / len(labels) for frequency in frequencies]
    entropy = 0

    # update the entropy according to the formula
    for probability in probabilities:
        if probability != 0:
            entropy -= probability * math.log2(probability)

    return entropy


def get_dataset_entropy(subset, dataset_size):
    """calculate the entropy of the subset

    :param subset: 2d array
    :param dataset_size: int
    """
    return len(subset) / dataset_size * calculate_dataset_entropy(subset)


def find_best_split(dataset, column_index):
    """find best split on a specific column

    :param dataset: matrix of input data,
    :param column index: int indicating the column
     """
    # initialise some variables
    best_split_value = None
    min_remainder = math.inf
    # split the dataset into right and left dataset
    partition_g = [[row[column_index], row[-1]] for row in dataset]
    partition_g.sort(key=lambda x: x[0])
    partition_le = []
    dataset_size = float(len(partition_g))

    # keep looping while there are still data in the right dataset
    while len(partition_g) > 1:
        # pop an item from right dataset and put into left dataset
        partition_le.append(partition_g.pop(0))

        # calculate the subset entropy
        if partition_le[-1][0] != partition_g[0][0]:
            remainder = get_dataset_entropy(partition_le, dataset_size) + \
                        get_dataset_entropy(partition_g, dataset_size)

            # if the remainder is smaller, update values
            if remainder < min_remainder:
                best_split_value = (partition_le[-1][0]
                                    + partition_g[0][0]) / 2
                min_remainder = remainder
                
    # return best split and the remainder
    return best_split_value, min_remainder


def find_split(dataset):
    """find_split

    :param dataset: matrix of input data, where each column except the last
    represents a feature and the last column represents the labels. """

    # initialise variables
    split_column_index = 0
    best_split_value = 0
    max_info_gain = -1
    # calculate dataset entropy according to the formula
    h_dataset = calculate_dataset_entropy(dataset)

    # loop through the columns
    for i in range(dataset.shape[1] - 1):
        # compute the best split on that column
        split_value, remainder = find_best_split(dataset, i)
        info_gain = h_dataset - remainder

        # if the info gain is greater, update the values
        if info_gain > max_info_gain:
            split_column_index = i
            best_split_value = split_value
            max_info_gain = info_gain

    # update left and right arrays based on the split
    matrix_le = np.array([row for row in dataset
                          if row[split_column_index] <= best_split_value])
    matrix_g = np.array([row for row in dataset
                         if row[split_column_index] > best_split_value])

    return (split_column_index,
            best_split_value,
            matrix_le,
            matrix_g)
