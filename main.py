import numpy as np

from decisiontrees import DecisionTreeClassifier, RandomForestClassifier
from decisiontrees.utils import k_folds_split


# np.random.seed(43010107)

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

    print(accuracy)
    print(average_cm)
    print(average_statistics)

    return accuracy, average_cm, average_statistics


if __name__ == "__main__":
    with open('data/clean_dataset.txt') as clean_dataset:
        data = np.loadtxt(clean_dataset)
    with open('data/noisy_dataset.txt') as noisy_dataset:
        noisy_data = np.loadtxt(noisy_dataset)
    np.random.shuffle(data)
    np.random.shuffle(noisy_data)

    rf_clean = RandomForestClassifier()
    rf_clean.fit(data[:1800])
    result_clean = rf_clean.evaluate(data[1800:])
    print(result_clean['accuracy'])

    rf_noisy = RandomForestClassifier()
    rf_noisy.fit(noisy_data[:1800])
    result_noisy = rf_noisy.evaluate(noisy_data[1800:])
    print(result_noisy['accuracy'])
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
