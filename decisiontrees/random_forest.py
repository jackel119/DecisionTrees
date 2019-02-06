from decisiontrees import DecisionTreeClassifier
from random import random, shuffle

import numpy as np
import pdb


class RandomForestClassifier:

    def __init__(self, n_trees=20):
        self.trees = []
        self.n_trees = n_trees

    def fit(self, train_data):
        """ Trains the random forest on the given train_data

        :param train_data: 2d numpy array with the last column as labels
        """

        no_features = train_data.shape[1]

        for _ in range(self.n_trees):
            np.random.shuffle(train_data)
            random_fraction = 0.4 + (random() * 0.4)
            random_subset = \
                train_data[:int(random_fraction * len(train_data))]

            random_fraction = 0.5 + (random() * 0.5)
            feature_index = list(range(no_features - 1))
            shuffle(feature_index)
            features_to_pick = \
                feature_index[: int(random_fraction
                                    * no_features)]
            features_to_pick.sort()
            features_to_pick.append(-1)

            random_subset = random_subset[:, features_to_pick]

            pdb.set_trace()

            tree = DecisionTreeClassifier()
            tree.fit(random_subset)

            self.trees.append((1, tree))

    def predict(self, x_data):
        """ Returns predictions of classes, given the x data

        :param x_data: unlabeled 2d numpy array
        """

        pass

    def evaluate(self, test_data):
        """ Returns a dictionary of statistics from an evaluation

        :param test_data: 2d numpy array with the last column as labels
        """

        return {
            "accuracy": 0, # TODO
            "confusion_matrix": 0, #TODO
            "stats": 0 # TODO
        }
