"""
This class is used to perform logistic regression classification
on NumPy arrays. It can similarly be used to perform multi-class
classification.
"""

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def fit(self, X, y, learning_rate=0.01, no_of_iterations=400, normalize=False,
            visualize=False, regularization=0.0):
        """
        Fits this model on the given datasets.

        Parameters:
        ___________
        X:  1D or 2D NumPy array of floats. If 1D, assumes X is an array with a single feature.
            If 2D, each column represents each feature and each row represents each test case.
        y:  1D NumPy array of floats representing the labels for each test case.
        learning_rate: Represents the step size on each iteration of gradient descent
            for how much to edit the model parameters (float value).
        no_of_iterations: Sets the number of iterations (int value) of gradient descent.
        normalize: Boolean determining whether to normalize features before fitting by
             subtracting the mean and dividing by standard deviation.
        visualize: If set to true, shows graphs of mean squared error vs. number of iterations
             as well as special graphs of the data if performing multi-class classification.
        regularization: A float that represents the regularization parameter used to prevent
            over-fitting. If set to 0, regularization is not performed.

        Returns:
        ________
        A NumPy array of NumPy arrays representing column vectors of the model parameters for
        each of the unique classifications for the data.
        """

        self.normalize_features = normalize

        if X.ndim == 1:
            X = X[np.newaxis].T

        temp_X = np.append(np.ones((X.shape[0], 1)), X, 1)
        if normalize:
            temp_X = np.append(np.ones((X.shape[0], 1)), self.normalize(X), 1)

        self.theta = self.gradient_descent(temp_X, y, learning_rate, no_of_iterations, visualize, regularization)

        return self.theta

    def gradient_descent(self, X, y, learning_rate=0.01, no_of_iterations=400, visualize=False, regularization=0.0):
        self.unique = np.unique(y)
        all_thetas = []
        temp_y = y
        for val in self.unique:
            theta = np.zeros((X.shape[1], 1))
            y = temp_y.copy()
            y[temp_y == val] = 1
            y[temp_y != val] = 0
            score = []
            score.append(self.cost(X, y, theta))
            m = X.shape[0]
            for i in range(no_of_iterations):
                partial = 1 / m * X.T.dot(self.sigmoid(X, theta) - y[np.newaxis].T)
                theta = theta * (1 - learning_rate * regularization / m) - learning_rate * partial
                score.append(self.cost(X, y, theta))
            all_thetas.append(theta)

            if visualize:
                fig, ax = plt.subplots()
                ax.plot(range(no_of_iterations + 1), score)
                ax.set_ylabel('Mean Squared Error')
                ax.set_xlabel('Number of Iterations')
                ax.set_title("Gradient descent when " + str(int(val)) + " label set to positive")
                plt.show()
        return np.asarray(all_thetas)

    def normalize(self, X):
        self.mean_of_features = []
        self.std_of_features = []
        temp_X = np.copy(X)
        for i in range(X.shape[1]):
            self.mean_of_features.append(np.mean(X[:, i]))
            self.std_of_features.append(np.std(X[:, i]))
            temp_X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
        return temp_X

    def sigmoid(self, X, theta):
        return 1 / (1 + np.e ** (-1 * X.dot(theta)))

    def cost(self, X, y, theta):
        return np.squeeze((y.dot(np.log(self.sigmoid(X, theta))) +
                (1 - y).dot(np.log(1 - self.sigmoid(X, theta))))) / (-1 * X.shape[0])

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
        A single value if one test case is passed in or a 1D NumPy array of the predicted
            values if a 2D array is passed in.
        """

        if X.ndim == 1:
            X = X[np.newaxis]
        if self.normalize_features:
            X = X.copy()
            for i in range(X.shape[1]):
                X[:, i] = (X[:, i] - self.mean_of_features[i]) / self.std_of_features[i]

        X = np.append(np.ones((X.shape[0], 1)), X, 1)
        predictions = []

        for thetas in self.theta:
            predictions.append(np.squeeze(self.sigmoid(X, thetas)))
        predictions = np.asarray(predictions)
        predictions = np.argmax(predictions, axis=0)
        predictions = self.unique[predictions]
        return predictions


def main():
    print('Single feature/binary classification logistic regression: ')
    test = np.loadtxt('data/logistic.txt', delimiter=',')
    X = test[:, 0]
    y = test[:, 1]

    lr = LogisticRegression()
    theta = lr.fit(X, y, visualize=True, learning_rate=0.003, no_of_iterations=6000)
    print('Parameter values: \n' + str(theta[1]))
    print('Predictions for 10, 50, and 100: ' + str(lr.predict(np.array([[10.], [50], [100]]))))

    plt.scatter(X, y, marker='.')
    plt.plot(X, lr.sigmoid(np.append(np.ones((X.shape[0], 1)), X[np.newaxis].T, 1), theta[1]))
    plt.title('Logistic Regression Prediction')
    plt.show()

    print('\n\nMultivariate/multi classification logistic regression: ')
    test = np.loadtxt('data/multi_classification.txt', delimiter=',')
    X = test[:, 0:2]
    y = test[:, 2]

    print('Add x1^x2, x1^2, and x2^2 as features to help predict better')
    X = np.append(X, (X[:, 0] * X[:, 1] / 100)[np.newaxis].T, axis=1)
    X = np.append(X, (X[:, 0] ** 2 / 100)[np.newaxis].T, axis=1)
    X = np.append(X, (X[:, 1] ** 2 / 100)[np.newaxis].T, axis=1)

    lr = LogisticRegression()
    theta = lr.fit(X, y, learning_rate=0.0001, no_of_iterations=1000)
    print('Predictions for (50, 50), (40, 40), (40, 60), (60, 30), and (60, 60)')
    print(lr.predict(np.array([[50., 50, 25, 25, 25],
                               [40., 40, 16, 16, 16],
                               [40., 60, 24, 16, 36],
                               [60., 30, 18, 36, 9],
                               [60., 60, 36, 36, 36]])))

    pred = lr.predict(X)
    fig, ax = plt.subplots()
    ax.scatter(X[pred == 0, 0], X[pred == 0,1], marker='o')
    ax.scatter(X[pred == 1, 0], X[pred == 1,1], marker='*')
    ax.scatter(X[pred == 2, 0], X[pred == 2,1], marker='+')
    ax.scatter(X[pred == 3, 0], X[pred == 3,1], marker='.')
    ax.set_title('Logistic Regression Predictions on Multi Training Set')
    plt.show()


if __name__ == '__main__':
    main()
