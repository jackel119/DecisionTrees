import numpy as np

from decisiontrees import DecisionTreeClassifier, RandomForestClassifier
from decisiontrees.utils import k_folds_split, split_data


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

def update_forest_val_acc(args):
    forest, val_acc_sum, train_data, validation_data = args
    forest.fit(train_data)
    statistics = forest.evaluate(validation_data)
    return val_acc_sum + statistics['accuracy']


def forest_k_folds_cv(dataset, k=10):
    accuracy_sum = 0
    cumulative_cm = np.zeros((4, 4))
    list_of_recalls = []
    list_of_precisions = []
    list_of_f1_measures = []
    # forest_sizes = [10, 20, 30, 35, 40, 50, 55, 65, 70, 75, 80, 85, 90, 100]
    forest_sizes = [10, 20, 30, 50]
    num_sizes = len(forest_sizes)
    forests = [RandomForestClassifier(size) for size in forest_sizes]

    # with Pool(k) as p:
    j = 0
    best_forest_sizes = []
    for train_validation_data, test_data in k_folds_split(dataset, k):
        val_acc_sums = [0] * num_sizes
        print(j)
        j += 1
        best_folds = [(0, i) for i in range(num_sizes)]
        for train_data, validation_data in \
                k_folds_split(train_validation_data, k - 1):
            # train_datas = [train_data] * num_sizes
            # validation_datas = [validation_data] * num_sizes
            # val_acc_sums = p.map(update_forest_val_acc, zip(forests, val_acc_sums, train_datas, validation_datas))
            for i in range(num_sizes):
                forest = forests[i]
                forest.fit(train_data)
                statistics = forest.evaluate(validation_data)
                val_acc_sums[i] += statistics['accuracy']
                if best_folds[i][0] < statistics['accuracy']:
                    best_folds[i] = (statistics['accuracy'], i)
        # print(test_data)
        print(val_acc_sums)
        (best_forest_idx, _) = max(list(enumerate(val_acc_sums)), key=lambda x: x[1])
        best_training_data, _ = split_data(train_validation_data, best_folds[best_forest_idx][1])
        best_forest = forests[best_forest_idx]
        best_forest.fit(best_training_data)
        statistics = best_forest.evaluate(test_data)
        accuracy_sum += statistics['accuracy']
        update_statistics(statistics, cumulative_cm, list_of_recalls,
                          list_of_f1_measures, list_of_precisions)
        best_forest_sizes.append(forest_sizes[best_forest_idx])
        print(accuracy_sum / j)

    accuracy = accuracy_sum / k
    average_cm = cumulative_cm / k
    calculate_average = lambda arr: [sum(x) / k for x in zip(*arr)]
    average_statistics = {'recalls': calculate_average(list_of_recalls),
                          'precisions': calculate_average(list_of_precisions),
                          'f1': calculate_average(list_of_f1_measures)}

    values, counts = np.unique(best_forest_sizes, return_counts=True)
    print(str(list(zip(values, counts))))
    print(max(zip(counts, values))[0])

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
