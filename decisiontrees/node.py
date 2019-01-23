import numpy as np
from decisiontrees.utils.find_split import find_split


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

        print("Generating Tree")
        classes = set(self.matrix[:,  -1])

        if len(classes) == 1:
            self.predict = lambda x: list(classes)[0]
        elif (self.matrix[:, 0: -1] == self.matrix[:, 0: -1][0]).all():
            labels = [int(label) for label in self.matrix[:, -1]]

            def predict(_):
                return np.random.choice(labels)
            self.predict = predict
        else:
            self._gen_nodes()

    def _gen_nodes(self):
        # Split the input x_matrix into left and right

        (split_func, left_matrix, right_matrix) = find_split(self.matrix)
        self.split_func = split_func
        self.left_node = Node(left_matrix)
        self.right_node = Node(right_matrix)

        def predict(row):
            if self.split_func(row):
                return self.right_node.predict(row)
            else:
                return self.left_node.predict(row)

        self.predict = predict
