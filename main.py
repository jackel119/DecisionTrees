import numpy as np

from decisiontrees import DecisionTreeClassifier

if __name__ == "__main__":
    with open('data/clean_dataset.txt') as clean_dataset:
        clean_data = np.loadtxt(clean_dataset)
    dt = DecisionTreeClassifier()
    dt.fit(clean_data)
    test_X, test_y = clean_data[:, :-1], clean_data[:, -1]
    pred_y = dt.predict(test_X)
    print(test_y)
    print(pred_y)
    results = np.column_stack((pred_y, test_y))
    print(dt)

    for result in results:
        print(result)
