from decisiontrees import DecisionTreeClassifier
from decisiontrees.utils import gen_quadrants_data


data = gen_quadrants_data(100)

dt = DecisionTreeClassifier()
dt.fit(data)
print(dt)
