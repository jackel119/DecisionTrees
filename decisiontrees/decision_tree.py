from node import Node

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
