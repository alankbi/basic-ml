"""
This class is an implementation of a neural network using the sigmoid
activation function.
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, input_nodes, output_nodes, hidden_layers, nodes_per_hidden_layer):
        """
        Create a NeuralNetwork with the given dimensions.

        Parameters:
        ___________
        input_nodes: Number of features each dataset will have.
        output_nodes: Number of unique labels to predict from the data.
        hidden_layers: Number of hidden layers (between the input and output layers).
        nodes_per_hidden_layer: Number of nodes per hidden layer (the same for all of them).
        """

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers
        self.nodes_per_hidden_layer = nodes_per_hidden_layer


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
        normalize: Boolean determining whether to normalize values before fitting by
             subtracting the mean and dividing by standard deviation.
        visualize: If set to true, shows graphs of mean squared error vs. number of iterations.
        regularization: A float that represents the regularization parameter used to prevent
            over-fitting. If set to 0, regularization is not performed.

        Returns:
        ________
        An array of 2D NumPy arrays. The NumPy array at index i represents the model parameters
        moving from layer i to layer i + 1 in the neural network.
        """

        self.normalize_features = normalize

        if X.ndim == 1:
            X = X[np.newaxis].T

        if normalize:
            X = self.normalize(X)

        self.theta = self.gradient_descent(X, y, learning_rate, no_of_iterations, visualize, regularization)

        return self.theta

    def gradient_descent(self, X, y, learning_rate=0.01, no_of_iterations=400, visualize=False, regularization=0.0):
        self.unique = np.unique(y)
        new_y = np.zeros((self.unique.size, y.size))
        for i, num in enumerate(self.unique):
            new_y[i, y == num] = 1

        # Random initialization of theta values
        theta = []
        if self.hidden_layers == 0:
            eps = 6 ** 0.5 / (self.input_nodes + self.output_nodes) ** 0.5
            theta.append(np.random.rand(self.output_nodes, self.input_nodes + 1) * 2 * eps - eps)
        else:
            eps = 6 ** 0.5 / (self.input_nodes + self.nodes_per_hidden_layer) ** 0.5
            theta.append(np.random.rand(self.nodes_per_hidden_layer, self.input_nodes + 1) * 2 * eps - eps)

        for i in range(self.hidden_layers):
            if i == self.hidden_layers - 1:
                eps = 6 ** 0.5 / (self.nodes_per_hidden_layer + self.output_nodes) ** 0.5
                theta.append(np.random.rand(self.output_nodes, self.nodes_per_hidden_layer + 1) * 2 * eps - eps)
            else:
                eps = 6 ** 0.5 / (self.nodes_per_hidden_layer * 2) ** 0.5
                theta.append(np.random.rand(self.nodes_per_hidden_layer,
                                                 self.nodes_per_hidden_layer + 1) * 2 * eps - eps)

        score = []
        for i in range(no_of_iterations):
            z, a = self.feed_forward(X.T, theta)
            h = a[len(a) - 1]

            cost = np.sum(-1 / len(y) * (new_y * np.log(h) + (1 - new_y) * np.log(1 - h))) # works
            reg = [np.sum(thetas ** 2) for thetas in theta]
            cost += regularization / (2 * len(y)) * sum(reg)
            score.append(cost)

            grad = []
            for j in range(len(theta)):
                grad.append(np.zeros(theta[j].shape))
            for j in range(len(y)):
                delta = h[:, j] - new_y[:, j]
                delta = delta[np.newaxis].T

                for k in range(len(grad) - 1, -1, -1):
                    grad[k] += delta.dot(np.append(1, a[k][:, j])[np.newaxis])
                    delta = theta[k].T.dot(delta) * (np.append(1, a[k][:, j]) * (1 - np.append(1, a[k][:, j])))[np.newaxis].T
                    delta = delta[1:]

            for j, gradient in enumerate(grad):
                gradient /= len(y)
                gradient += regularization / len(y) * gradient
                theta[j] -= learning_rate * gradient

            """ # GRADIENT CHECKING
            import copy
            temp_grad = []
            for j in range(len(theta)):
                temp_grad.append(np.zeros(theta[j].shape))
            for j in range(len(theta)):
                for (k, l), val in np.ndenumerate(temp_grad[j]):
                    theta_plus = copy.deepcopy(theta)
                    theta_plus[j][k, l] += 10 ** -4
                    theta_minus = copy.deepcopy(theta)
                    theta_minus[j][k, l] -= 10 ** -4

                    # Cost for theta_plus
                    z, a = self.feed_forward(X.T, theta_plus)
                    h = a[len(a) - 1]
                    cost = np.sum(-1 / len(y) * (new_y * np.log(h) + (1 - new_y) * np.log(1 - h)))
                    temp_grad[j][k, l] = cost
                    # Cost for theta_minus
                    z, a = self.feed_forward(X.T, theta_minus)
                    h = a[len(a) - 1]
                    temp_grad[j][k, l] -= np.sum(-1 / len(y) * (new_y * np.log(h) + (1 - new_y) * np.log(1 - h)))
                    temp_grad[j][k, l] /= 2 * 10 ** -4

            # Compare backpropagation and gradient checking side by side
            for i in range(len(grad)):
                print(np.append(grad[i], temp_grad[i], axis=1))"""

        if visualize:
            plt.plot(range(len(score)), score)
            plt.ylabel('Mean Squared Error')
            plt.xlabel('Number of Iterations')
            plt.show()

        return theta

    def normalize(self, X):
        self.mean_of_features = []
        self.std_of_features = []
        temp_X = np.copy(X)
        for i in range(X.shape[1]):
            self.mean_of_features.append(np.mean(X[:, i]))
            self.std_of_features.append(np.std(X[:, i]))
            temp_X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
        return temp_X

    def sigmoid(self, X):
        return 1 / (1 + np.e ** (-1 * X))

    def feed_forward(self, X, theta):
        z = []
        a = []

        z.append(np.zeros(X.shape))  # To keep axis aligned with a
        a.append(X)

        for i in range(self.hidden_layers + 1):
            z.append(theta[i].dot(np.append(np.ones((1, a[i].shape[1])), a[i], axis=0)))
            a.append(self.sigmoid(z[i + 1]))
        return z, a

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
        A 1D NumPy array of the predicted labels for each test case.
        """

        if X.ndim == 1:
            X = X[np.newaxis]
        if self.normalize_features:
            X = X.copy()
            for i in range(X.shape[1]):
                X[:, i] = (X[:, i] - self.mean_of_features[i]) / self.std_of_features[i]

        __, predictions = self.feed_forward(X.T, self.theta)
        predictions = np.argmax(predictions[len(predictions) - 1], axis=0)
        predictions = self.unique[predictions]
        return predictions


# Example code:

print("Neural network: ")
test = np.loadtxt('data/multi_classification.txt', delimiter=',')
X = test[:, 0:2]
y = test[:, 2]
X = np.append(X, (X[:, 0] * X[:, 1] / 100)[np.newaxis].T, axis=1)
X = np.append(X, (X[:, 0] ** 2 / 100)[np.newaxis].T, axis=1)
X = np.append(X, (X[:, 1] ** 2 / 100)[np.newaxis].T, axis=1)

fig, ax = plt.subplots()
ax.scatter(X[y == 0, 0], X[y == 0, 1], marker='o')
ax.scatter(X[y == 1, 0], X[y == 1, 1], marker='*')
ax.scatter(X[y == 2, 0], X[y == 2, 1], marker='+')
ax.scatter(X[y == 3, 0], X[y == 3, 1], marker='.')
plt.show()

nn = NeuralNetwork(5, 4, 1, 8)
nn.fit(X, y, no_of_iterations=1000, learning_rate=0.05, visualize=True)
print("Predictions for (50, 50), (40, 40), (40, 60), (60, 30), and (60, 60)")
print(nn.predict(np.array([[50., 50, 25, 25, 25],
                           [40., 40, 16, 16, 16],
                           [40., 60, 24, 16, 36],
                           [60., 30, 18, 36, 9],
                           [60., 60, 36, 36,36]])))

pred = nn.predict(X)
fig, ax = plt.subplots()
ax.scatter(X[pred == 0, 0], X[pred == 0, 1], marker='o')
ax.scatter(X[pred == 1, 0], X[pred == 1, 1], marker='*')
ax.scatter(X[pred == 2, 0], X[pred == 2, 1], marker='+')
ax.scatter(X[pred == 3, 0], X[pred == 3, 1], marker='.')
ax.set_title("Neural Network Predictions on Training Set")
plt.show()
