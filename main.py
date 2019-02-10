import numpy as np

from decisiontrees.evaluate import k_folds_cv

if __name__ == "__main__":
    # open file containing data
    with open('data/noisy_dataset.txt') as f:
        data = np.loadtxt(f)
    # randomly shuffle data
    np.random.seed(50)
    np.random.shuffle(data)
    # compute the statistics of the k fold Evaluation
    # of the algorithm
    evaluation =\
        k_folds_cv(data, k=10, validation=True)
    print(evaluation["accuracy"])
    print(evaluation["confusion_matrix"])
    print(evaluation["statistics"])
