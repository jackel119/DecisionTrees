from decisiontrees import DecisionTreeClassifier
from random import random, shuffle

import numpy as np
import pdb

from decisiontrees.utils import build_confusion_matrix, stats


class RandomForestClassifier:

    def __init__(self, n_trees=50):
        self.forest_info = []
        self.n_trees = n_trees
        self.num_labels = 0

    def fit(self, train_data):
        """ Trains the random forest on the given train_data

        :param train_data: 2d numpy array with the last column as labels
        """

        self.num_labels = int(max(train_data[:, -1]))

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

            # pdb.set_trace()

            tree = DecisionTreeClassifier()
            tree.fit(random_subset)

            accuracy = tree.evaluate(train_data)["accuracy"]

            self.forest_info.append((accuracy, tree, features_to_pick[:-1]))

    def predict(self, x_data):
        """ Returns predictions of classes, given the x data

        :param x_data: unlabeled 2d numpy array
        """
        num_rows = len(x_data)
        predictions = np.zeros((num_rows, self.num_labels))
        for (vote_weight, tree, features) in self.forest_info:
            predicted_labels = tree.predict(x_data[:, features])
            for i in range(num_rows):
                predictions[i, int(predicted_labels[i] - 1)] += vote_weight
        # print(predictions)
        result = [np.argmax(prediction) + 1 for prediction in predictions]
        return np.array(result)

    def evaluate(self, test_data):
        """ Returns a dictionary of statistics from an evaluation

        :param test_data: 2d numpy array with the last column as labels
        """

        test_X, test_y = test_data[:, :-1], test_data[:, -1]
        pred_y = self.predict(test_X)
        cm = build_confusion_matrix(pred_y, test_y)
        print(cm)
        correct_predictions = 0
        for i in range(len(cm)):
            correct_predictions += cm[i, i]

        return {
            "accuracy": correct_predictions / len(test_data),
            "confusion_matrix": cm,
            "stats": stats(cm)
        }
