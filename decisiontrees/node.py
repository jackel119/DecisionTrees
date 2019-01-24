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

    def __repr__(self):
        if self.is_leaf:
            return "Leaf " + str(self.predict(None))
        else:
            return "Node Col " + str(self.split_col) +\
                " Value " + str(self.split_value) + " (" +\
                self.left_node.__repr__() + ") (" \
                + self.right_node.__repr__() + ")"
