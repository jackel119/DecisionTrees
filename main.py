import numpy as np

from decisiontrees import DecisionTreeClassifier, RandomForestClassifier
from decisiontrees.utils import k_folds_split


def build_tree(train_data, validation_data=None, pruning=False):
    dt = DecisionTreeClassifier()
    dt.fit(train_data)

    if pruning:
        if validation_data is None:
            raise Exception("Cannot prune without validation data!")
        dt.prune(validation_data)

    return dt


def update_statistics(statistics, cumulative_cm, list_of_recalls,
                      list_of_f1_measures, list_of_precisions):
    cumulative_cm += statistics['confusion_matrix']
    list_of_recalls.append(statistics['stats']['recalls'])
    list_of_precisions.append(statistics['stats']['precisions'])
    list_of_f1_measures.append(statistics['stats']['f1'])


def k_folds_cv(dataset, k=10, validation=False):
    accuracy_sum = 0
    cumulative_cm = np.zeros((4, 4))
    list_of_recalls = []
    list_of_precisions = []
    list_of_f1_measures = []

    if validation:
        num_samples = k * (k - 1)
    else:
        num_samples = k

    for train_validation_data, test_data in k_folds_split(dataset, k):
        if not validation:
            dt = build_tree(train_validation_data)
            statistics = dt.evaluate(test_data)
            accuracy_sum += statistics['accuracy']
            update_statistics(statistics, cumulative_cm, list_of_recalls,
                              list_of_f1_measures, list_of_precisions)
        else:
            for train_data, validation_data in \
                    k_folds_split(train_validation_data, k - 1):
                dt = build_tree(train_data, validation_data, pruning=True)
                statistics = dt.evaluate(test_data)
                accuracy_sum += statistics['accuracy']
                update_statistics(statistics, cumulative_cm, list_of_recalls,
                                  list_of_f1_measures, list_of_precisions)

    accuracy = accuracy_sum / num_samples
    average_cm = cumulative_cm / num_samples
    calculate_average = lambda arr: [sum(x) / num_samples for x in zip(*arr)]
    average_statistics = {'recalls': calculate_average(list_of_recalls),
                          'precisions': calculate_average(list_of_precisions),
                          'f1': calculate_average(list_of_f1_measures)}

    return {
        "accuracy": accuracy,
        "confusion_matrix": average_cm,
        "statistics": average_statistics
    }


if __name__ == "__main__":
    with open('data/clean_dataset.txt') as f:
        data = np.loadtxt(f)
    np.random.seed(50)
    np.random.shuffle(data)
    evaluation =\
        k_folds_cv(data, k=10, validation=True)
    print(evaluation["accuracy"])
    print(evaluation["confusion_matrix"])
    print(evaluation["statistics"])
