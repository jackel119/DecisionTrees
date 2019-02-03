from decisiontrees.node import Node
from decisiontrees.utils import build_confusion_matrix, stats


import matplotlib.pyplot as plt

import numpy as np


class DecisionTreeClassifier:

    def __init__(self):
        pass

    def fit(self, train_data):
        """ Trains the tree on the given train_data

        :param train_data: 2d numpy array with the last column as labels
        """
        self.root_node = Node(train_data)

    def predict(self, x_data):
        """ Returns predictions of classes, given the x data

        :param x_data: unlabeled 2d numpy array
        """
        result = []

        for row in x_data:
            result.append(self.root_node.predict(row))

        return np.array(result)

    def evaluate(self, test_data):
        """ Returns a dictionary of statistics from an evaluation

        :param test_data: 2d numpy array with the last column as labels
        """
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
        """ Prunes the tree, given a set of validation data to prune with

        :param prune_data: 2d numpy array with the last column as labels
        :param debug: Enable debugging printing
        """

        if len(prune_data) == 0:
            raise Exception("Validation data is empty!")
        self.root_node.prune(prune_data, debug=debug)

    def _plot_tree_util(self, is_left, node, x1, y1, x2, y2):
        midx = (x1+x2)/2
        plt.text(midx, y2, str(node), size=10, color='white',
                 ha="center", va="center",
                 bbox=dict(facecolor='black', edgecolor='red'))

        if is_left:
            parentx = x2
        else:
            parentx = x1

        # Draw line to parent
        plt.plot([parentx, midx], [y2+10, y2], 'ro-')

        if not node.is_leaf:
            self._plot_tree_util(True, node.left_node,
                                 x1, y1, (x1+x2)/2, y2-10)
            self._plot_tree_util(False, node.right_node,
                                 (x1+x2)/2, y1, x2, y2-10)

    def plot_tree(self):
        """Plots the tree using matplotlib"""
        # Plot root
        x1 = 0
        y1 = 0
        x2 = 10000
        y2 = 1000
        midx = (x1+x2)/2
        plt.text(midx, y2, str(self.root_node), size=10, color='white',
                 ha="center", va="center",
                 bbox=dict(facecolor='black', edgecolor='red'))

        self._plot_tree_util(True, self.root_node.left_node,
                             x1, y1, midx, y2-10)
        self._plot_tree_util(False, self.root_node.right_node,
                             midx, y1, x2, y2-10)
        plt.show()

    def height(self):
        """ Height of the tree"""

        return self.root_node._height()

    def average_height(self):
        """ Average Height of the tree """

        return self.root_node._average_height()

    def __len__(self):
        return self.height()

    def __repr__(self):
        return self.root_node.__repr__()
