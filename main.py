import numpy as np

from decisiontrees.evaluate import k_folds_cv

if __name__ == "__main__":
    with open('data/noisy_dataset.txt') as f:
        data = np.loadtxt(f)
    np.random.seed(50)
    np.random.shuffle(data)
    evaluation =\
        k_folds_cv(data, k=10, validation=True)
    print(evaluation["accuracy"])
    print(evaluation["confusion_matrix"])
    print(evaluation["statistics"])
