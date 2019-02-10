import numpy as np
from decisiontrees.utils import find_split


class Node:
    def __init__(self, matrix):
        """ Constructor

        :param matrix: matrix of training data, with labels in the format
        [f1, f2, f3....., label]
        """

        # data that is passed at this node
        self.matrix = matrix
        self.most_frequent_label = -1
        self._rand_dist = False
        self._gen_tree()
        pass

    def _gen_tree(self):
        """ Generate trees recursively
        (if there is a sensible split)"""

        # get the list of lables for the data at this node
        labels = [int(label) for label in self.matrix[:, -1]]
        # count each lable's occurance and find the most frequent label
        training_frequencies = np.bincount(labels)
        self.most_frequent_label = np.argmax(training_frequencies)

        # If all the labels are the same, this is a leaf
        if training_frequencies[self.most_frequent_label] == len(labels):
            # set the predict function to predict this lable
            self.predict = lambda _: self.most_frequent_label
            # set leaf variable
            self.is_leaf = True

        # no sensible split can be generated when different entries
        # that share the same feature have different labels
        elif (self.matrix[:, 0: -1] == self.matrix[0, 0: -1]).all():

            # use a weighted random distribution as the prediction functions
            self.rand_labels = [int(label) for label in self.matrix[:, -1]]
            self.rand_labels.sort()

            def predict(_):
                return np.random.choice(self.rand_labels)

            self.predict = predict
            self._rand_dist = True
            self.is_leaf = True

        else:
            # otherwwise it can be further splitted
            self.is_leaf = False
            # generate children nodes
            self._gen_nodes()

    def _gen_nodes(self):
        """ Generates left and right nodes """

        # find sensible split using the util function
        (split_col_index, split_value,
         left_matrix, right_matrix) = find_split(self.matrix)

        # set variables
        self.split_col = split_col_index
        self.split_value = split_value
        # define split function to be based on the split column and value
        self.split_func = lambda data: data[split_col_index] <= split_value
        # generate left and right nodes
        self.left_node = Node(left_matrix)
        self.right_node = Node(right_matrix)

        # define predict function to be called on the left or right node
        # that will be decided by the split function
        def predict(row):
            if self.split_func(row):
                return self.left_node.predict(row)
            else:
                return self.right_node.predict(row)

        self.predict = predict

    def prune(self, validation_data, debug=False):
        """prune

        :param validation_data: matrix of validation data
        :param debug: enable debugging console prints
        """

        # if node is leaf and no validation data is passed to this node
        # delete this node
        if not self.is_leaf:
            if len(validation_data) == 0:
                self._prune_replacement(debug=debug)

                return

            # if left node isn't leaf, prune on the left node
            if not self.left_node.is_leaf:
                left_matrix = np.array([row for row in validation_data
                                        if self.split_func(row)])

                self.left_node.prune(left_matrix, debug=debug)

            # if right node isn't leaf, prune on right node
            if not self.right_node.is_leaf:
                right_matrix = np.array([row for row in validation_data
                                         if not self.split_func(row)])

                self.right_node.prune(right_matrix, debug=debug)

            # check if right/left nodes are leaves after pruning
            if self.left_node.is_leaf and self.right_node.is_leaf:
                # calculate no prune accuracy on this node
                no_replacement_acc = self.evaluate(validation_data)
                # check the frequency of most frequent label in validation
                # data that is passed to this node
                freq_in_validation_data = \
                    len(validation_data[validation_data[:, -1] ==
                                        self.most_frequent_label])
                # calculate accuracy for pruning using the frequency
                replacement_acc = \
                    freq_in_validation_data / len(validation_data)

                # if prune accuracy is not worse than no prune accuracy
                # prune this node
                if replacement_acc >= no_replacement_acc:
                    self._prune_replacement(debug=debug)

        return

    def _prune_replacement(self, debug=False):
        """ Replace current node with leaf of most_freq_label

        :param debug: Bool, Debug printing
        """

        if debug:
            print("Replacing", repr(self),
                  "with (Leaf, ", self.most_frequent_label)
        # update prediction function to predict the most freqent label
        # at this node, and remove left, right nodes, set to leaf node
        self.predict = lambda _: self.most_frequent_label
        self.is_leaf = True
        self.left_node = None
        self.right_node = None

        return

    def evaluate(self, test_data):
        """ Evaluation function

        :param test_data: labeled test matrix
        :return: accuracy as a float
        """
        # split the test data into features and labels
        test_X, test_y = test_data[:, :-1], test_data[:, -1]
        # map the prediction on the features
        pred_y = np.array(list(map(self.predict, test_X)))
        # calculate accuracy by checking the number of correctly predicted
        # labels
        accuracy = np.sum(test_y == pred_y) / len(test_y)

        return accuracy

    def _height(self):
        """ calculate the height of the node recursively """
        if self.is_leaf:
            return 0
        else:
            return 1 + max(self.left_node._height(),
                           self.right_node._height())

    def _average_height(self):
        """ calculate the average height of the node recursively """
        if self.is_leaf:
            return 1
        else:
            return 1 + ((self.left_node._average_height() +
                         self.right_node._average_height()) / 2)

    def __str__(self):
        """ function use to represent the node in the plot """
        if not self.is_leaf:
            return ("x" + str(self.split_col) + "<=" + str(self.split_value))
        else:
            return str(self.predict(None))

    def __repr__(self):
        """ function for debugging purposes """
        if self.is_leaf:
            if self._rand_dist:
                return "Rand Dist " + str(list(set(self._rand_dist)))
            else:
                return "Leaf " + str(self.predict(None))
        else:
            return "Node Col " + str(self.split_col) + \
                   " Value " + str(self.split_value) + " (" + \
                   self.left_node.__repr__() + ") (" \
                   + self.right_node.__repr__() + ")"
