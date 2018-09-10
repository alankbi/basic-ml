"""
This class is used to perform decision tree classification on
NumPy arrays.
"""

import numpy as np
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

        return self.build_tree(X, y)

    def build_tree(self, X, y):
        return None

    def find_split(self, X, y):
        return None

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

        return None


class DecisionTreeNode:

    def __init__(self, rule, true_branch=None, false_branch=None):
        self.rule = rule
        self.true_branch = true_branch
        self.false_branch = false_branch


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

# Test gini_impurity method
tree = DecisionTree()
impurity = tree.gini_impurity(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
print(impurity)

# Test match and partition
rule = Rule(3, 1)
foo = rule.match(np.array([[1, 2, 3],
                           [4, 5, 6]]))
print(foo)
print(tree.partition(np.array([[1, 2, 3],
                              [4, 5, 6]]), np.array([1, 2]), rule))

# Test information gain
print(tree.information_gain(np.array([1, 1, 1]), np.array([1]), np.array([1, 1])))
print(tree.information_gain(np.array([2, 2, 3, 3]), np.array([2, 2]), np.array([3, 3])))

