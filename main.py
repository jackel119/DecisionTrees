import numpy as np

with open('data/clean_dataset.txt') as clean_dataset:
    data = np.loadtxt(clean_dataset)
    print(data)
