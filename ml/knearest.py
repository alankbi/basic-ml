"""
This class is used to perform K-Nearest Neighbors classification
on NumPy arrays with Euclidean distance. It can similarly be
used to perform multi-class classification.
"""

import numpy as np
import matplotlib.pyplot as plt


class KNearestNeighbors:

    def fit(self, X, y, normalize=False):
        """
        Fits this model on the given datasets.

        Parameters:
        ___________
        X:  1D or 2D NumPy array of floats. If 1D, assumes X is an array with a single feature.
            If 2D, each column represents each feature and each row represents each test case.
        y:  1D NumPy array of floats representing the labels for each test case.
        normalize: Boolean determining whether to normalize features before fitting by
             subtracting the mean and dividing by standard deviation.
        """

        self.normalize_features = normalize

        if X.ndim == 1:
            X = X[np.newaxis].T

        if normalize:
            X = self.normalize(X)

        self.X = X
        self.y = y.astype(int)

    def normalize(self, X):
        self.mean_of_features = []
        self.std_of_features = []
        temp_X = np.copy(X)
        for i in range(X.shape[1]):
            self.mean_of_features.append(np.mean(X[:, i]))
            self.std_of_features.append(np.std(X[:, i]))
            temp_X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
        return temp_X

    def distance(self, x1, x2):
        return np.sum((x2 - x1) ** 2, axis=1) ** 0.5

    def predict(self, X, K=3):
        """
        Predicts the label from the given input features.

        Parameters:
        ___________
        X:  1D or 2D NumPy array of floats. If 1D, then the array is treated as one test
            with each index being its value for that different feature. If 2D, each row
            is a test case and each column is a feature.

        Returns:
        ________
        A 1D NumPy array of the predicted labels for the given data.
        """

        if X.ndim == 1:
            X = X[np.newaxis]
        if self.normalize_features:
            X = X.copy()
            for i in range(X.shape[1]):
                X[:, i] = (X[:, i] - self.mean_of_features[i]) / self.std_of_features[i]

        predictions = []

        for i in range(X.shape[0]):
            distances = self.distance(X[i, :], self.X)
            closest = distances.argsort()[:K]
            frequencies = np.bincount(self.y[closest])
            predictions.append(np.argmax(frequencies))

        return np.array(predictions)


def main():
    print('K-Nearest Neighbors: ')
    test = np.loadtxt('data/multi_classification.txt', delimiter=',')
    X = test[:, 0:2]
    y = test[:, 2]

    fig, ax = plt.subplots()
    ax.scatter(X[y == 0, 0], X[y == 0, 1], marker='o')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], marker='*')
    ax.scatter(X[y == 2, 0], X[y == 2, 1], marker='+')
    ax.scatter(X[y == 3, 0], X[y == 3, 1], marker='.')
    plt.show()

    knn = KNearestNeighbors()
    knn.fit(X, y)
    print('Predictions for (50, 50), (40, 40), (40, 60), (60, 30), and (60, 60)')
    print(knn.predict(np.array([[50., 50],
                                [40., 40],
                                [40., 60],
                                [60., 30],
                                [60., 60]])))

    pred = knn.predict(X, K=5)
    fig, ax = plt.subplots()
    ax.scatter(X[pred == 0, 0], X[pred == 0, 1], marker='o')
    ax.scatter(X[pred == 1, 0], X[pred == 1, 1], marker='*')
    ax.scatter(X[pred == 2, 0], X[pred == 2, 1], marker='+')
    ax.scatter(X[pred == 3, 0], X[pred == 3, 1], marker='.')
    ax.set_title('K-Nearest Neighbors Predictions on Training Set')
    plt.show()


if __name__ == '__main__':
    main()
