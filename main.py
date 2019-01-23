import numpy as np
from decisiontrees.decision_tree import DecisionTreeClassifier

if __name__ == "__main__":
    with open('data/clean_dataset.txt') as clean_dataset:
        clean_data = np.loadtxt(clean_dataset)
    dt = DecisionTreeClassifier().fit(clean_data)
