import numpy as np
np.random.seed(42)

from decisiontrees import DecisionTreeClassifier

if __name__ == "__main__":
    with open('data/noisy_dataset.txt') as clean_dataset:
        clean_data = np.loadtxt(clean_dataset)
    np.random.shuffle(clean_data)
    dt = DecisionTreeClassifier()
    train_size = 1600
    train_data = clean_data[:train_size]
    dt.fit(train_data)
    test_X, test_y = clean_data[train_size:, :-1], clean_data[train_size:, -1]
    pred_y = dt.predict(test_X)
    results = np.column_stack((pred_y, test_y))
    print(dt)
    print(test_y == pred_y)
    print("Accuracy: ", np.sum(test_y == pred_y) / len(test_y))
