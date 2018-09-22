"""
This class is used to perform decision tree classification on
NumPy arrays.
"""

import numpy as np
import matplotlib.pyplot as plt
from numbers import Number


class DecisionTree:

    def fit(self, X, y):
        """
        Fits this model on the given datasets.

        Parameters:
        ___________
        X:  1D or 2D NumPy array of floats. If 1D, assumes X is an array with a single feature.
            If 2D, each column represents each feature and each row represents each test case.
        y:  1D NumPy array of floats representing the labels for each test case.

        Returns:
        ________
        A DecisionTreeNode representing the root node of the decision tree.
        """

        if X.ndim == 1:
            X = X[np.newaxis].T

        self.root = self.build_tree(X, y)
        return self.root

    def build_tree(self, X, y):
        rule, gain = self.find_split(X, y)
        if gain == 0:
            return LeafNode(y)
        true_branch_X, true_branch_y, false_branch_X, false_branch_y = self.partition(X, y, rule)
        return DecisionNode(rule, self.build_tree(true_branch_X, true_branch_y),
                            self.build_tree(false_branch_X, false_branch_y))

    def find_split(self, X, y):
        best_gain = 0
        best_rule = None

        for i in range(X.shape[1]):
            unique_values = np.unique(X[:, i])
            for value in unique_values:
                rule = Rule(value, i)
                __, true_branch_y, __, false_branch_y = self.partition(X, y, rule)
                gain = self.information_gain(y, true_branch_y, false_branch_y)
                if gain > best_gain:
                    best_gain = gain
                    best_rule = rule

        return best_rule, best_gain

    def gini_impurity(self, labels):
        impurity = 1.0
        __, counts = np.unique(labels, return_counts=True)
        for count in counts:
            impurity -= (1.0 * count / len(labels)) ** 2
        return impurity

    def information_gain(self, original, true_branch, false_branch):
        total_labels = len(true_branch) + len(false_branch)
        weighted_gini_impurity = (len(true_branch) * self.gini_impurity(true_branch) +
                                  len(false_branch) * self.gini_impurity(false_branch)) / total_labels
        return self.gini_impurity(original) - weighted_gini_impurity

    def partition(self, X, y, rule):
        true_branch_X = X[rule.match(X), :]
        true_branch_y = y[rule.match(X)]
        false_branch_X = X[~rule.match(X), :]
        false_branch_y = y[~rule.match(X)]

        return true_branch_X, true_branch_y, false_branch_X, false_branch_y

    def predict(self, X):
        """
        Predicts the label from the given input features.

        Parameters:
        ___________
        X:  1D or 2D NumPy array of floats. If 1D, then the array is treated as one test
            with each index being its value for that different feature. If 2D, each row
            is a test case and each column is a feature.

        Returns:
        ________
        A single float if one test case is passed in or a 1D NumPy array of the predicted
            values if a 2D array is passed in.
        """

        if X.ndim == 1:
            X = X[np.newaxis]

        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = self.root
            while isinstance(node, DecisionNode):
                if node.rule.match(X[i, :])[0]:  # Get single boolean value
                    node = node.true_branch
                else:
                    node = node.false_branch
            predictions[i] = node.label

        return np.squeeze(predictions)


class DecisionNode:

    def __init__(self, rule, true_branch, false_branch):
        self.rule = rule
        self.true_branch = true_branch
        self.false_branch = false_branch


class LeafNode:

    def __init__(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        self.label = unique[np.argmax(counts)]


class Rule:

    def __init__(self, value, column):
        self.value = value
        self.column = column

    def match(self, X):
        if X.ndim == 1:
            X = X[np.newaxis]
        if isinstance(self.value, Number):
            return X[:, self.column] >= self.value
        return X[:, self.column] == self.value


# Example code:

print("Decision Tree: ")
test = np.loadtxt('data/multi_classification.txt', delimiter=',')
X = test[:, 0:2]
y = test[:, 2]

fig, ax = plt.subplots()
ax.scatter(X[y == 0, 0], X[y == 0, 1], marker='o')
ax.scatter(X[y == 1, 0], X[y == 1, 1], marker='*')
ax.scatter(X[y == 2, 0], X[y == 2, 1], marker='+')
ax.scatter(X[y == 3, 0], X[y == 3, 1], marker='.')
plt.show()

tree = DecisionTree()
tree.fit(X, y)
print("Predictions for (50, 50), (40, 40), (40, 60), (60, 30), and (60, 60)")
predictions = tree.predict(np.array([[50., 50],
                                     [40., 40],
                                     [40., 60],
                                     [60., 30],
                                     [60., 60]]))
print(predictions)

pred = tree.predict(X)
fig, ax = plt.subplots()
ax.scatter(X[pred == 0, 0], X[pred == 0, 1], marker='o')
ax.scatter(X[pred == 1, 0], X[pred == 1, 1], marker='*')
ax.scatter(X[pred == 2, 0], X[pred == 2, 1], marker='+')
ax.scatter(X[pred == 3, 0], X[pred == 3, 1], marker='.')
ax.set_title("Decision Tree Predictions on Training Set")
plt.show()