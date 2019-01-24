from decisiontrees.utils import build_confusion_matrix
from decisiontrees import DecisionTreeClassifier

import numpy as np
np.random.seed(50)


def validate_model(train_data, test_data, print_confusion_matrix=False,
                   pruning=False):
    dt = DecisionTreeClassifier()
    dt.fit(train_data)
    acc = dt.evaluate(test_data)

    # if print_confusion_matrix:
    #     confusion_matrix = build_confusion_matrix(pred_y, test_y)
    #     print(confusion_matrix)

    pruned_acc = -1

    if pruning:
        dt.prune(test_data)
        pruned_acc = dt.evaluate(test_data)

    print(acc, pruned_acc)


def k_folds_cv(dataset, k=10):
    partition_size = len(dataset) // k
    accuracy_sum = 0

    for i in range(k):
        test_data = dataset[i * partition_size: (i + 1) * partition_size]
        train_indexes = np.r_[0: (i * partition_size),
                              ((i + 1) * partition_size): len(dataset)]
        train_data = dataset[train_indexes]
        accuracy_sum += validate_model(train_data, test_data)
    print(accuracy_sum / k)


if __name__ == "__main__":
    with open('data/clean_dataset.txt') as clean_dataset:
        data = np.loadtxt(clean_dataset)
    with open('data/noisy_dataset.txt') as noisy_dataset:
        noisy_data = np.loadtxt(noisy_dataset)
    np.random.shuffle(data)
    np.random.shuffle(noisy_data)
    # k_folds_cv(data)
    # k_folds_cv(noisy_data)
    validate_model(data[:, :1600], data[:, 1600:], pruning=True)
