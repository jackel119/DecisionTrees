# Decision Trees

This is a simple implementation of decision trees.


The interface is inspired by other popular machine learning libraries such as `Keras` and `SKLearn`, and hence the main `DecisionTreeClassifier` class has methods `fit`, `predict`, `evaluate`.

### Instantiation and training

To use, first import (from top level of this folder, similar to what can be found in `main.py`)

```python
from decisiontrees import DecisionTreeClassifier
dt = DecisionTreeClassifier()
```

Then, simply `fit` the model on some data:

```python
dt.fit(training_data)
```

The input data format must be a 2-dimensional `numpy` `ndarray`, with the last column being integer labels. For example, with the given datasets in the coursework, you can do:

```python
with open('data/clean_dataset.txt') as clean_dataset:
  training_data = np.loadtxt(clean_dataset)
```

### Making predictions and evaluations

To make a prediction, simply call `predict`:

```python
dt.predict(test_data)
```
where `test_data` is the exact same format as the training data, but without the last column (of labels).

There is also an `evaluate` function:

```python
dt.evaluate(labeled_test_data)
```
where `labeled_test_data` is the same format as `training_data` (i.e. it is labeled). What this will do is internally call `predict` on the features, then compare with the actual labels in order to compute accuracy, precision and recall (of each class), as well as a confusion matrix (as a `numpy` array, with predictions as columns and actual labels as rows). This is all returned as a dictionary.

For example:

```python
>>> evaluation = dt.evaluate(labeled_test_data)
>>> evaluation['accuracy']
0.975
>>> evaluation['stats']
{
  'recalls': [1.0, 1.0, 0.9878048780487805, 1.0],
  'precisions': [1.0, 0.9857142857142858, 1.0, 1.0],
  'f1': [1.0, 0.9928057553956835, 0.9938650306748467, 1.0]
}
>>> evaluation['confusion_matrix']
array([[79,  0,  0,  0],
       [ 0, 69,  0,  0],
       [ 0,  1, 81,  0],
       [ 0,  0,  0, 70]])
```

Note that the lists of `stats` are indexed by their class label minus one, i.e. the value at the 0th index of the `precision` list is the precision of class 1.

### Tree Representations

You can also visualize the tree, using `dt.plot_tree()`, which will give you something like this:

![Tree Example](images/tree.png)
