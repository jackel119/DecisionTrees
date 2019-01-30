from decisiontrees.node import Node
from decisiontrees.utils import build_confusion_matrix
import matplotlib.pyplot as plt


import numpy as np


class DecisionTreeClassifier:

    def __init__(self, n_layers=None):
        pass

    def fit(self, train_data):
        self.root_node = Node(train_data)

    def predict(self, x_data):
        result = []

        for row in x_data:
            result.append(self.root_node.predict(row))

        return np.array(result)

    def evaluate(self, test_data):
        test_X, test_y = test_data[:, :-1], test_data[:, -1]
        pred_y = self.predict(test_X)
        cm = build_confusion_matrix(pred_y, test_y)
        acc = self.root_node.evaluate(test_data)

        return {
            "accuracy": acc,
            "confusion_matrix": cm
            #"stats": stats(cm)
        }

    def prune(self, prune_data, debug=False):
        self.root_node.prune(prune_data, debug=debug)
        
    def print2DUtil(self, node, space):
        # if node is None:
        #     return;
        
        space += 10
        
        if not node.is_leaf:
            self.print2DUtil(node.right_node, space)
        
        print("\n")
        for i in range(10, space):
            print(" ", end='')
        if not node.is_leaf:
            print("x" + str(node.split_col) + "<=" + str(node.split_value))
        else:
            print(node.predict(None))
        
        if not node.is_leaf:
            self.print2DUtil(node.left_node, space)
        
    def print2D(self):
        self.print2DUtil(self.root_node, 0);
        
    def plotTreeUtil(self, isLeft, node, x1, y1, x2, y2):
        midx = (x1+x2)/2
        plt.text(midx, y2, str(node), size=10, color='white',
            ha="center", va="center",
            bbox=dict(facecolor='black', edgecolor='red'))
        
        if isLeft:
            parentx = x2
        else:
            parentx = x1
        
        #draw line to parent
        plt.plot([parentx, midx], [y2+10, y2], 'ro-')
        if not node.is_leaf:
            self.plotTreeUtil(True, node.left_node, x1, y1, (x1+x2)/2, y2-10)
            self.plotTreeUtil(False, node.right_node, (x1+x2)/2, y1, x2, y2-10)
        
    def plotTree(self):
        #plot root
        x1 = 0
        y1 = 0
        x2 = 10000
        y2 = 1000
        midx = (x1+x2)/2
        plt.text(midx, y2, str(self.root_node), size=10, color='white',
            ha="center", va="center",
            bbox=dict(facecolor='black', edgecolor='red'))

        #check leaf later?
        self.plotTreeUtil(True, self.root_node.left_node, x1, y1, midx, y2-10)
        self.plotTreeUtil(False, self.root_node.right_node, midx, y1, x2, y2-10)
        plt.show()
    
            
    def __repr__(self):
        return self.root_node.__repr__()
        
        
