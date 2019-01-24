import numpy as np


def build_confusion_matrix(pred_y, actual_y):
    confusion_matrix_size = int(max(max(pred_y), max(actual_y)))

    confusion_matrix = [[0 for _ in range(confusion_matrix_size)]
                        for _ in range(confusion_matrix_size)]
    for x, y in zip(pred_y, actual_y):
        confusion_matrix[int(x) - 1][int(y) - 1] += 1
    return np.array(confusion_matrix)
