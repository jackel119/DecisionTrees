import math

NUM_ROOMS = 4


def calculate_dataset_entropy(dataset):
    labels = [row[-1] for row in dataset]
    frequencies = [0] * NUM_ROOMS
    for label in labels:
        frequencies[label - 1] += 1
    probabilities = [frequency / len(labels) for frequency in frequencies]
    entropy = 0
    for probability in probabilities:
        entropy -= probability * math.log2(probability)
    return entropy


def get_dataset_entropy(subset, dataset_size):
    return len(subset) / dataset_size * calculate_dataset_entropy(subset)


def find_best_split(dataset, column_index):
    best_split_value = None
    min_remainder = math.inf
    partition_g = [[row[column_index], row[-1]] for row in dataset]
    partition_g.sort(key=lambda x: x[0])
    partition_le = []
    dataset_size = float(len(partition_g))
    while len(partition_g) > 1:
        partition_le.append(partition_g.pop(0))
        if partition_le[-1][1] != partition_g[0][1]:
            remainder = get_dataset_entropy(partition_le, dataset_size) + \
                        get_dataset_entropy(partition_g, dataset_size)
            if remainder < min_remainder:
                best_split_value = (partition_le[-1][0] + partition_g[0][0]) / 2
    return best_split_value, min_remainder


def find_split(dataset):
    """find_split

    :param dataset: matrix of input data, where each column except the last
    represents a feature and the last column represents the labels. """

    split_column_index = 0
    best_split_value = 0
    max_info_gain = -1
    h_dataset = calculate_dataset_entropy(dataset)

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
