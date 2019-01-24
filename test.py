from decisiontrees import DecisionTreeClassifier
from decisiontrees.utils import gen_quadrants_data

import numpy as np


data = gen_quadrants_data(1000)

X, y = data[:, :-1], data[:, -1]


dt = DecisionTreeClassifier()
dt.fit(data)
print(dt)


data = gen_quadrants_data(100)

X, y = data[:, :-1], data[:, -1]

pred_y = dt.predict(X)
print(np.column_stack((pred_y, y)))
