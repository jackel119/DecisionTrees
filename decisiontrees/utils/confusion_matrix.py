import numpy as np


def build_confusion_matrix(pred_y, actual_y):
    """build confusion matrix from predictions and actual labels

    :param pred_y: np array of labels
    :param actual_y: np array of labels
    """
    # compute the size of the matrix
    confusion_matrix_size = int(max(max(pred_y), max(actual_y)))

    # initialise confusion matrix to matrix of zeros
    confusion_matrix = [[0 for _ in range(confusion_matrix_size)]
                        for _ in range(confusion_matrix_size)]

    # for each pair, update correct position of the matrix
    for x, y in zip(actual_y, pred_y):
        confusion_matrix[int(x) - 1][int(y) - 1] += 1

    return np.array(confusion_matrix)


def stats(cm):
    """compute statistics using confusion matrix

    :param cm: 2d array
    """
    # initialise the statistics
    statistics = {}
    recalls = []
    precisions = []
    f1_measures = []
    accuracy_sum = 0
    num_entries = 0
    for i in range(len(cm)):
        accuracy_sum += cm[i, i]
        tp = cm[i, i]
        row = cm[i]
        col = cm[:, i]

        row_sum = sum(row)
        num_entries += row_sum

        if row_sum == 0:
            recall = 1
        else:
            recall = tp / row_sum

        col_sum = sum(col)
        if col_sum == 0:
            precision = 1
        else:
            precision = tp / col_sum

        if tp == 0:
            f1_measure = 1
        else:
            f1_numerator = (recall * precision)
            f1_denominator = (recall + precision)
            f1_measure = 2 * (f1_numerator / f1_denominator)

        recalls.append(recall)
        precisions.append(precision)
        f1_measures.append(f1_measure)
    statistics['accuracy'] = accuracy_sum / num_entries
    statistics['recalls'] = recalls
    statistics['precisions'] = precisions
    statistics['f1'] = f1_measures

    return statistics
