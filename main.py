from decisiontrees import DecisionTreeClassifier
from decisiontrees.utils import build_confusion_matrix, \
    split_data, k_folds_split

import numpy as np
np.random.seed(50)


def validate_model(train_data, test_data, validation_data=None,
                   print_confusion_matrix=False,
                   pruning=False, debug=False):
    dt = DecisionTreeClassifier()
    dt.fit(train_data)
    acc = dt.evaluate(test_data)['accuracy']
    dt.plotTree()

    # if print_confusion_matrix:
    #     confusion_matrix = build_confusion_matrix(pred_y, test_y)
    #     print(confusion_matrix)

    if pruning:
        if validation_data is None:
            raise Exception("Cannot prune without validation data!")
        dt.prune(validation_data, debug=debug)
        pruned_acc = dt.evaluate(test_data)['accuracy']
        # print(acc, pruned_acc)

        return pruned_acc
    else:
        # print(acc)

        return acc


def k_folds_cv(dataset, k=10, validation=False):
    accuracy_sum = 0

    for train_validation_data, test_data in k_folds_split(dataset, k):
        if not validation:
            accuracy_sum += validate_model(train_validation_data, test_data)
        else:
            fold_acc_sum = 0

            for train_data, validation_data in \
                    k_folds_split(train_validation_data, k - 1):
                fold_acc_sum += validate_model(train_data,
                                               test_data,
                                               validation_data,
                                               pruning=True)
            accuracy_sum += (fold_acc_sum / (k-1))

    print(accuracy_sum / k)


if __name__ == "__main__":
    with open('data/clean_dataset.txt') as clean_dataset:
        data = np.loadtxt(clean_dataset)
    with open('data/noisy_dataset.txt') as noisy_dataset:
        noisy_data = np.loadtxt(noisy_dataset)
    np.random.shuffle(data)
    np.random.shuffle(noisy_data)
    # k_folds_cv(data, validation=True)
    # k_folds_cv(noisy_data, validation=True)
    res = validate_model(data[:1600], data[1800:],
                         validation_data=data[1600:1800],
                         pruning=True)
    print(res)
