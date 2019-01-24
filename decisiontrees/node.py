import numpy as np
from decisiontrees.utils import find_split


class Node:
    def __init__(self, matrix):
        """__init__

        :param matrix: matrix of training data, with labels in the format
        [f1, f2, f3....., label]
        """

        self.matrix = matrix
        self._gen_tree()
        pass

    def _gen_tree(self):

        classes = set(self.matrix[:,  -1])

        if len(classes) == 1:

            self.predict = lambda x: list(classes)[0]
            self.is_leaf = True

        elif (self.matrix[:, 0: -1] == self.matrix[:, 0: -1][0]).all():

            labels = [int(label) for label in self.matrix[:, -1]]

            def predict(_):
                return np.random.choice(labels)
            self.predict = predict
            self.is_leaf = True

        else:
            self.is_leaf = False
            self._gen_nodes()

    def _gen_nodes(self):
        # Split the input x_matrix into left and right

        (split_col_index, split_value,
         left_matrix, right_matrix) = find_split(self.matrix)

        self.split_col = split_col_index
        self.split_value = split_value
        self.split_func = lambda data: data[split_col_index] <= split_value
        self.left_node = Node(left_matrix)
        self.right_node = Node(right_matrix)

        def predict(row):
            if self.split_func(row):
                return self.left_node.predict(row)
            else:
                return self.right_node.predict(row)

        self.predict = predict

    def prune(self, validation_data, debug=False):

        if not self.is_leaf:
            # First prune left

            if not self.left_node.is_leaf:
                left_matrix = np.array([row for row in validation_data
                                        if self.split_func(row)])

                if len(left_matrix) > 0:
                    self.left_node.prune(left_matrix, debug=debug)

            if not self.right_node.is_leaf:
                right_matrix = np.array([row for row in validation_data
                                        if not self.split_func(row)])

                if len(right_matrix) > 0:
                    self.right_node.prune(right_matrix, debug=debug)

            # And then condition check if both left and right are leaves
            # after pruning

            if self.left_node.is_leaf and self.right_node.is_leaf:

                # If two leaves are the same, then always replace and halt

                if self.left_node.predict(None) \
                        == self.right_node.predict(None):
                    self._prune_replacement(self.left_node, debug=debug)

                    return

                # Two leaves are different,
                # check if pruning will improve accuracy
                no_replacement_acc = self.evaluate(validation_data)
                left_replacement_acc = self.left_node.evaluate(validation_data)
                right_replacement_acc = \
                    self.right_node.evaluate(validation_data)

                if no_replacement_acc > left_replacement_acc \
                        and no_replacement_acc > right_replacement_acc:
                    # We don't prune
                    pass
                else:
                    if left_replacement_acc >= right_replacement_acc \
                            and left_replacement_acc >= no_replacement_acc:
                        # Replace with left node
                        self._prune_replacement(self.left_node, debug=debug)
                    else:
                        # Replace with right node
                        self._prune_replacement(self.right_node, debug=debug)

    def _prune_replacement(self, replacement_node, debug=False):

        if debug:
            print("Replacing", str(self),
                  "with", str(replacement_node))
        self.predict = replacement_node.predict

        self.left_node = None
        self.right_node = None
        self.is_leaf = True

    def evaluate(self, test_data):
        test_X, test_y = test_data[:, :-1], test_data[:, -1]
        pred_y = np.array(list(map(self.predict, test_X)))
        accuracy = np.sum(test_y == pred_y) / len(test_y)

        return accuracy

    def __repr__(self):
        if self.is_leaf:
            return "Leaf " + str(self.predict(None))
        else:
            return "Node Col " + str(self.split_col) +\
                " Value " + str(self.split_value) + " (" +\
                self.left_node.__repr__() + ") (" \
                + self.right_node.__repr__() + ")"
