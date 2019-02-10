import numpy as np

from decisiontrees import DecisionTreeClassifier
from decisiontrees.utils import build_confusion_matrix, stats

from random import random, shuffle


class RandomForestClassifier:

    def __init__(self, n_trees=20):
        """ initialise the random forest

        :param n_trees: int, default is 20:
        """
        self.forest_info = []
        self.n_trees = n_trees
        self.num_labels = 0

    def fit(self, train_data):
        """ Trains the random forest on the given train_data

        :param train_data: 2d numpy array with the last column as labels
        """

        # update the number of labels
        self.num_labels = int(max(train_data[:, -1]))

        # update the number of features
        no_features = train_data.shape[1]

        # loop through the number of trees
        for _ in range(self.n_trees):
            # shuffle the training data
            np.random.shuffle(train_data)
            # take a random fraction of the data
            random_fraction = 0.4 + (random() * 0.4)
            random_subset = \
                train_data[:int(random_fraction * len(train_data))]

            # take a proportion of the features
            random_fraction = 0.5 + (random() * 0.5)
            feature_index = list(range(no_features - 1))
            shuffle(feature_index)
            features_to_pick = \
                feature_index[: int(random_fraction
                                    * no_features)]
            features_to_pick.sort()
            features_to_pick.append(-1)

            random_subset = random_subset[:, features_to_pick]

            # build the tree on and fit it on the subset of the data created
            tree = DecisionTreeClassifier()
            tree.fit(random_subset)

            # compute the accuracy of the tree on the training data
            accuracy = tree.evaluate(train_data)["accuracy"]

            # append all these info into the forest
            # accuracy is used as the weight of voting
            self.forest_info.append((accuracy, tree, features_to_pick[:-1]))

    def predict(self, x_data):
        """ Returns predictions of classes, given the x data

        :param x_data: unlabeled 2d numpy array
        """
        # number of rows of data to predict
        num_rows = len(x_data)
        # initialise predictions
        predictions = np.zeros((num_rows, self.num_labels))

        # loop through the trees
        for (vote_weight, tree, features) in self.forest_info:
            # predict the labels using the tree
            predicted_labels = tree.predict(x_data[:, features])

            # update the weighted votes for labels
            for i in range(num_rows):
                predictions[i, int(predicted_labels[i] - 1)] += vote_weight

        # take the labels with the most votes for each data entry
        result = [np.argmax(prediction) + 1 for prediction in predictions]

        return np.array(result)

    def evaluate(self, test_data):
        """ Returns a dictionary of statistics from an evaluation

        :param test_data: 2d numpy array with the last column as labels
        """

        # split the data into features and labels
        test_X, test_y = test_data[:, :-1], test_data[:, -1]
        # predict the features
        pred_y = self.predict(test_X)
        # compute confusion matrix
        cm = build_confusion_matrix(pred_y, test_y)
        correct_predictions = 0

        # update the accuracy on based on the diagonal values
        for i in range(len(cm)):
            correct_predictions += cm[i, i]

        return {
            "accuracy": correct_predictions / len(test_data),
            "confusion_matrix": cm,
            "stats": stats(cm)
        }
