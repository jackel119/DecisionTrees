import numpy as np

from decisiontrees import DecisionTreeClassifier

if __name__ == "__main__":
    with open('data/clean_dataset.txt') as clean_dataset:
        clean_data = np.loadtxt(clean_dataset)
    dt = DecisionTreeClassifier()
    dt.fit(clean_data)
    test_set = clean_data[:, :-1]
    print(test_set)
    print(set(dt.predict(test_set)))
