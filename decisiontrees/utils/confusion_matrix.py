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
        recall = tp / sum(row)
        precision = tp / sum(col)
        f1_measure = 2 * ((recall * precision) / (recall + precision))
        recalls.append(recall)
        precisions.append(precision)
        f1_measures.append(f1_measure)
    statistics["recalls"] = recalls
    statistics["precisions"] = precisions
    statistics["F1-measures"] = f1_measures
    return statistics
