import numpy as np


def build_confusion_matrix(pred_y, actual_y):
    confusion_matrix_size = int(max(max(pred_y), max(actual_y)))

    confusion_matrix = [[0 for _ in range(confusion_matrix_size)]
                        for _ in range(confusion_matrix_size)]

    for x, y in zip(actual_y, pred_y):
        confusion_matrix[int(x) - 1][int(y) - 1] += 1

    return np.array(confusion_matrix)


def stats(cm):
    statistics = {}
    recalls = []
    precisions = []
    f1_measures = []

    for i in range(len(cm)):
        tp = cm[i, i]
        row = cm[i]
        col = cm[:, i]

        row_sum = sum(row)
        if row_sum == 0:
            recall = 0
        else:
            recall = tp / row_sum

        col_sum = sum(col)
        if col_sum == 0:
            precision = 0
        else:
            precision = tp / col_sum

        if recall == 0 and precision == 0:
            f1_measure = -1
        else:
            f1_numerator = (recall * precision)
            f1_denominator = (recall + precision)
            f1_measure = 2 * (f1_numerator / f1_denominator)

        recalls.append(recall)
        precisions.append(precision)
        f1_measures.append(f1_measure)
    statistics['recalls'] = recalls
    statistics['precisions'] = precisions
    statistics['f1'] = f1_measures

    return statistics
