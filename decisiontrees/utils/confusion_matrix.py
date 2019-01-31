import numpy as np


def build_confusion_matrix(pred_y, actual_y):
    confusion_matrix_size = int(max(max(pred_y), max(actual_y)))

    confusion_matrix = [[0 for _ in range(confusion_matrix_size)]
                        for _ in range(confusion_matrix_size)]

    for x, y in zip(pred_y, actual_y):
        confusion_matrix[int(x) - 1][int(y) - 1] += 1

    return np.array(confusion_matrix)


def stats(cm):
    # TODO: Try and optimise this function to iterate over the confusion
    #  matrix once
    statistics = {}
    recalls = []
    precisions = []
    f1_measures = []
    for i in range(len(cm)):
        tp = 0
        fn = 0
        for j in range(len(cm)):
            if i == j:
                tp += cm[i][j]
            else:
                fn += cm[i][j]
        recalls.append(tp / (tp + fn))
    statistics["recalls"] = recalls
    for i in range(len(cm)):
        tp = 0
        fp = 0
        for j in range(len(cm)):
            if i == j:
                tp += cm[j][i]
            else:
                fp += cm[j][i]
        precisions.append(tp / (tp + fp))
    statistics["precisions"] = precisions
    for i in range(len(recalls)):
        recall = recalls[i]
        precision = precisions[i]
        f1_measures.append(2 * ((recall * precision) / (recall + precision)))
    statistics["F1-measures"] = f1_measures
    return statistics
