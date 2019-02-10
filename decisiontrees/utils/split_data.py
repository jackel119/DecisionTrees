import numpy as np


def split_data(dataset, i, k=10):
    """ Splits data at index i

    :param dataset:
    :param i: int
    :param k: int
    """
    partition_size = len(dataset) // k

    # split the data at index i
    test_data = dataset[i * partition_size: (i + 1) * partition_size]
    train_indexes = np.r_[0: (i * partition_size),
                          ((i + 1) * partition_size): len(dataset)]
    train_data = dataset[train_indexes]

    return train_data, test_data


def k_folds_split(dataset, k):
    """ Splits data into partitions for k folds

    :param dataset:
    :param k: int
    """
    for i in range(k):
        yield split_data(dataset, i, k)
