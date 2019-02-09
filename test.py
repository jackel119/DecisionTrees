from decisiontrees import DecisionTreeClassifier, RandomForestClassifier
from decisiontrees.utils import gen_quadrants_data

import numpy as np


data = gen_quadrants_data(1000)

X, y = data[:, :-1], data[:, -1]


dt = DecisionTreeClassifier()
dt.fit(data)
print(dt)


data = gen_quadrants_data(1000)

X, y = data[:, :-1], data[:, -1]

test_data = gen_quadrants_data(300)
__import__('pdb').set_trace()
