import numpy as np

from decisiontrees import DecisionTreeClassifier
from decisiontrees.utils import \
    k_folds_split, stats

# np.random.seed(50)


# def validate_model(train_data, test_data, validation_data=None,
#                    stats=False,
#                    pruning=False, debug=False):
#     dt = build_tree(train_data, validation_data, pruning)
#     dt_statistics = dt.evaluate(test_data)
#
#     if stats:
#         print(dt_statistics["confusion_matrix"])
#         print(dt_statistics["stats"]["recalls"])
#         print(dt_statistics["stats"]["precisions"])
#         print(dt_statistics["stats"]["F1-measures"])
#
#     return dt_statistics['accuracy']


def build_tree(train_data, validation_data=None, pruning=False):
    dt = DecisionTreeClassifier()
    dt.fit(train_data)

    if pruning:
        if validation_data is None:
            raise Exception("Cannot prune without validation data!")
        dt.prune(validation_data)

    return dt


def k_folds_cv(dataset, k=10, validation=False):
    accuracy_sum = 0
    cumulative_cm = np.zeros((4, 4))
    if validation:
        num_samples = k * (k - 1)
    else:
        num_samples = k

    for train_validation_data, test_data in k_folds_split(dataset, k):
        if not validation:
            dt = build_tree(train_validation_data)
            statistics = dt.evaluate(test_data)
            accuracy_sum += statistics['accuracy']
            cumulative_cm += statistics['confusion_matrix']
        else:
            for train_data, validation_data in \
                    k_folds_split(train_validation_data, k - 1):
                dt = build_tree(train_data, validation_data, pruning=True)
                statistics = dt.evaluate(test_data)
                accuracy_sum += statistics['accuracy']
                cumulative_cm += statistics['confusion_matrix']

    cumulative_cm /= num_samples
    average_statistics = stats(cumulative_cm)

    print(accuracy_sum / num_samples)
    print(cumulative_cm)
    print(average_statistics)

    return accuracy_sum / num_samples, cumulative_cm, average_statistics


if __name__ == "__main__":
    with open('data/clean_dataset.txt') as clean_dataset:
        data = np.loadtxt(clean_dataset)
    with open('data/noisy_dataset.txt') as noisy_dataset:
        noisy_data = np.loadtxt(noisy_dataset)
    np.random.shuffle(data)
    np.random.shuffle(noisy_data)
    # k_folds_cv(data, validation=True)
    # k_folds_cv(noisy_data, validation=True)
    # k_folds_cv(noisy_data, validation=False)
    # k_folds_cv(data, validation=True)
    # k_folds_cv(data, validation=False)
    # res = validate_model(noisy_data[:1600], noisy_data[1800:],
    #                      validation_data=noisy_data[1600:1800],
    #                      pruning=True, stats=True)
    # res1 = validate_model(noisy_data[:1800], noisy_data[1800:],
    #                       pruning=False, stats=True)
    # print(res, res1)
    # validate_model(data[:1800], data[1800:], stats=True)
