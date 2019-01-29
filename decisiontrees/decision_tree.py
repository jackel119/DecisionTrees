from decisiontrees.node import Node
from decisiontrees.utils import build_confusion_matrix, stats

import numpy as np


class DecisionTreeClassifier:

    def __init__(self, n_layers=None):
        pass

    def fit(self, train_data):
        self.root_node = Node(train_data)

    def predict(self, x_data):
        result = []

        for row in x_data:
            result.append(self.root_node.predict(row))

        return np.array(result)

    def evaluate(self, test_data):
        test_X, test_y = test_data[:, :-1], test_data[:, -1]
        pred_y = self.predict(test_X)
        cm = build_confusion_matrix(pred_y, test_y)
        acc = self.root_node.evaluate(test_data)

        return {
            "accuracy": acc,
            "confusion_matrix": cm,
            "stats": stats(cm)
        }

    def prune(self, prune_data, debug=False):
        self.root_node.prune(prune_data, debug=debug)

    def __repr__(self):
        return self.root_node.__repr__()
