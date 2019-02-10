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

        # loop through rows in the test data and call predict function on them,
        # then append the prediction results to a output array
        for row in x_data:
            result.append(self.root_node.predict(row))

        return np.array(result)

    def evaluate(self, test_data):
        """ Returns a dictionary of statistics from an evaluation

        :param test_data: 2d numpy array with the last column as labels
        """
        # split test_data into prediction data and test lables
        test_X, test_y = test_data[:, :-1], test_data[:, -1]
        # predict test data
        pred_y = self.predict(test_X)
        # build confusion matrix with prediction results and actual lables
        cm = build_confusion_matrix(pred_y, test_y)
        # call evaluate function to return the accuracy of the prediction
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

        # unable to prune with validation data
        if len(prune_data) == 0:
            raise Exception("Validation data is empty!")
        # call root's prune function
        self.root_node.prune(prune_data, debug=debug)

    def _plot_tree_util(self, parentx, node, x1, y1, x2, y2):
        # calculate mid point of the sub window
        midx = (x1+x2)/2
        # plot node
        plt.text(midx, y2, str(node), size=10, color='white',
                 ha="center", va="center",
                 bbox=dict(facecolor='black', edgecolor='red'))

        # Draw line to parent
        # simple because the distance between node and children is 10
        plt.plot([parentx, midx], [y2+10, y2], 'ro-')

        # if the node still has children, call this fucntion recursively
        if not node.is_leaf:
            # get the height of the left and right node
            left_height = node.left_node._height() + 1
            right_height = node.right_node._height() + 1
            # update the weight value to be proportion of left height over
            # total height
            weight = left_height / (left_height + right_height)
            # use this weight to divide the mid point of its children
            # the children node with a higher height will get assigned
            # a slightly larger window
            div_x = x1 + weight * (x2 - x1)
            # plot child nodes
            self._plot_tree_util(midx, node.left_node,
                                 x1, y1, div_x, y2 - 10)
            self._plot_tree_util(midx, node.right_node,
                                 div_x, y1, x2, y2-10)

    def plot_tree(self):
        """Plots the tree using matplotlib"""
        # Plot root
        # set arbitrary window size, width (x1 to x2) and height (y1 to y2)
        x1 = 0
        y1 = 0
        x2 = 10000
        y2 = 1000
        # midpoint of the window to plot root
        midx = (x1+x2)/2
        # plot root node as a rectangle
        plt.text(midx, y2, str(self.root_node), size=10, color='white',
                 ha="center", va="center",
                 bbox=dict(facecolor='black', edgecolor='red'))

        # call helper functions on left and right node to plot the children
        # in the subwindows divided by midpoint
        # define the vertical distance between node and its children
        # to be an arbitrary value of 10
        self._plot_tree_util(midx, self.root_node.left_node,
                             x1, y1, midx, y2-10)
        self._plot_tree_util(midx, self.root_node.right_node,
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
