import numpy as np

from decisiontrees import DecisionTreeClassifier
from decisiontrees.utils import k_folds_split


def build_tree(train_data, validation_data=None, pruning=False):
    """ build a decision tree

    :param train_data: 2d np array, validation data 2d np array, pruning: Bool
    """

    # Build a decision tree and by fitting the training data
    dt = DecisionTreeClassifier()
    dt.fit(train_data)

    # if pruning and validation data is existent, prune the decision tree
    # with validation data
    if pruning:
        if validation_data is None:
            raise Exception("Cannot prune without validation data!")
        dt.prune(validation_data)

    return dt


def update_statistics(statistics, cumulative_cm, list_of_recalls,
                      list_of_f1_measures, list_of_precisions):
    """ util function to update the statistics

    :param statistics: dictionary, cumulative_cm: matrix (2d np array),
    list_of_recalls: list of list; list_of_f1_measures: list of list,
    list_of_precisions: list of list
    """
    cumulative_cm += statistics['confusion_matrix']
    list_of_recalls.append(statistics['stats']['recalls'])
    list_of_precisions.append(statistics['stats']['precisions'])
    list_of_f1_measures.append(statistics['stats']['f1'])


def k_folds_cv(dataset, k=10, validation=False):
    # cumulative acuracy and cumulative confusion matrix defined
    accuracy_sum = 0
    cumulative_cm = np.zeros((4, 4))
    # list of precision, recall, and f1 to be updated and averaged upon
    list_of_recalls = []
    list_of_precisions = []
    list_of_f1_measures = []

    # different method is used for cases with validation data (pruning) or not
    if validation:
        num_samples = k * (k - 1)
    else:
        num_samples = k

    # split the data into training(train validation data) and test data
    # into k number of k-1 : 1 splits
    for train_validation_data, test_data in k_folds_split(dataset, k):
        # if no pruning in the tree
        if not validation:
            # build the tree
            dt = build_tree(train_validation_data)
            # compute statistics
            statistics = dt.evaluate(test_data)
            # accumulate accuracy
            accuracy_sum += statistics['accuracy']
            # update statistics
            update_statistics(statistics, cumulative_cm, list_of_recalls,
                              list_of_f1_measures, list_of_precisions)
        # pruning is used in the tree
        else:
            # another split for training data and validation data into
            # k-1 number of k-2 : 1 splits
            for train_data, validation_data in \
                    k_folds_split(train_validation_data, k - 1):
                # build the decision tree and prune on the validation data
                dt = build_tree(train_data, validation_data, pruning=True)
                # compute statistics
                statistics = dt.evaluate(test_data)
                # acccumulate accuracy
                accuracy_sum += statistics['accuracy']
                # update the statistics
                update_statistics(statistics, cumulative_cm, list_of_recalls,
                                  list_of_f1_measures, list_of_precisions)

    # compute average statistics for all of the metrics
    accuracy = accuracy_sum / num_samples
    average_cm = cumulative_cm / num_samples
    calculate_average = lambda arr: [sum(x) / num_samples for x in zip(*arr)]
    average_statistics = {'recalls': calculate_average(list_of_recalls),
                          'precisions': calculate_average(list_of_precisions),
                          'f1': calculate_average(list_of_f1_measures)}

    # return the statistics as a distionary
    return {
        "accuracy": accuracy,
        "confusion_matrix": average_cm,
        "statistics": average_statistics
    }
