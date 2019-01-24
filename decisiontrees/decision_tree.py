from decisiontrees.node import Node

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
        return self.root_node.evaluate(test_data)

    def prune(self, prune_data, debug=False):
        self.root_node.prune(prune_data, debug=debug)

    def __repr__(self):
        return self.root_node.__repr__()
