"""
This class is used to perform linear regression on NumPy arrays.
The model parameters can be fit using either gradient descent or
the normal equation.
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def fit(self, X, y, use_gradient_descent=True, learning_rate=0.01, no_of_iterations=400,
            normalize=False, visualize=False, regularization=0.0):
        """
        Fits this model on the given datasets.

        Parameters:
        ___________
        X:  1D or 2D NumPy array of floats. If 1D, assumes X is an array with a single feature.
            If 2D, each column represents each feature and each row represents each test case.
        y:  1D NumPy array of floats representing the labels for each test case.
        use_gradient_descent: If set to true, fits model with gradient descent. Otherwise,
            uses the normal equation.
        learning_rate: If using gradient descent, represents the step size on each iteration
            for how much to edit the model parameters (float value).
        no_of_iterations: If using gradient descent, sets the number of iterations (int value).
        normalize: If using gradient descent, this boolean determines whether to normalize
            values before fitting (subtract the mean and divide by standard deviation).
        visualize: If set to true and using gradient descent, shows a graph of mean squared
            error vs. number of iterations.
        regularization: A float that represents the regularization parameter used to prevent
            over-fitting. If set to 0, regularization is not performed.

        Returns:
        ________
        A 2D NumPy array representing a column vector of the model parameters.
        """

        self.use_gradient_descent = use_gradient_descent
        self.normalize_features = normalize

        if X.ndim == 1:
            X = X[np.newaxis].T

        temp_X = np.append(np.ones((X.shape[0], 1)), X, 1)
        if use_gradient_descent:
            if normalize:
                temp_X = np.append(np.ones((X.shape[0], 1)), self.normalize(X), 1)
            self.theta = self.gradient_descent(temp_X, y, learning_rate, no_of_iterations, visualize, regularization)
        else:
            self.theta = self.normal_equation(temp_X, y, regularization)

        return self.theta

    def gradient_descent(self, X, y, learning_rate=0.01, no_of_iterations=400, visualize=False, regularization=0.0):
        theta = np.zeros((X.shape[1], 1))
        score = []
        score.append(self.cost(X, y, theta))
        m = X.shape[0]
        for i in range(no_of_iterations):
            partial = 1 / m * (X.dot(theta) - y[np.newaxis].T).T.dot(X)
            theta = theta * (1 - learning_rate * regularization / m) - learning_rate * partial.T
            score.append(self.cost(X, y, theta))

        if visualize:
            plt.plot(range(no_of_iterations + 1), score)
            plt.ylabel('Mean Squared Error')
            plt.xlabel('Number of Iterations')
            plt.show()
        return theta

    def normal_equation(self, X, y, regularization=0.0):
        return np.linalg.inv(X.T.dot(X) + regularization * np.eye(X.shape[1])).dot(X.T).dot(y)[np.newaxis].T

    def normalize(self, X):
        self.mean_of_features = []
        self.std_of_features = []
        temp_X = np.copy(X)
        for i in range(X.shape[1]):
            self.mean_of_features.append(np.mean(X[:, i]))
            self.std_of_features.append(np.std(X[:, i]))
            temp_X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
        return temp_X

    def cost(self, X, y, theta):
        return np.sum((X.dot(theta) - y[np.newaxis].T) ** 2) / (2 * X.shape[0])

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
        if self.use_gradient_descent and self.normalize_features:
            X = X.copy()
            for i in range(X.shape[1]):
                X[:, i] = (X[:, i] - self.mean_of_features[i]) / self.std_of_features[i]

        return np.squeeze(np.append(np.ones((X.shape[0], 1)), X, 1).dot(self.theta))


# Example code:

print("Single feature linear regression: ")
test = np.loadtxt('data/linear.txt', delimiter=',')
X = test[:, 0]
y = test[:, 1]

lr = LinearRegression()
theta = lr.fit(X, y, visualize=True, normalize=True)
print("Gradient descent: ")
print("Parameter values: \n" + str(theta))
print("Predictions for 10, 50, and 100: " + str(lr.predict(np.array([[10.], [50], [100]]))))

print("\nNormal equation: ")
theta = lr.fit(X, y, use_gradient_descent=False)
print("Predictions for 10, 50, and 100: " + str(lr.predict(np.array([[10.], [50], [100]]))))

plt.plot(X, y, X, theta[0] + theta[1] * X)
plt.title("Linear Regression Prediction")
plt.show()


print("\n\nMultiple feature linear regression: ")
test = np.loadtxt('data/linear_multi.txt', delimiter=',')
X = test[:, 0:2]
y = test[:, 2]

print("Gradient descent: ")
lr = LinearRegression()
theta = lr.fit(X, y, visualize=True, no_of_iterations=1000, normalize=True)
print("Parameter values: \n" + str(theta))
print("Predictions for (20, 30) and (50, 70): " + str(lr.predict(np.array([[20., 30], [50., 70]]))))

print("\nNormal equation: ")
theta = lr.fit(X, y, use_gradient_descent=False, regularization=1)
print("Parameter values: \n" + str(theta))
print("Predictions for (20, 30) and (50, 70): " + str(lr.predict(np.array([[20., 30], [50., 70]]))))

help(lr.fit)
help(lr.predict)
