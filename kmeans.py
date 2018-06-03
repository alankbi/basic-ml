"""
This class is used to perform logistic regression classification
on NumPy arrays. It can similarly be used to perform multi-class
classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from sys import maxsize
from random import sample


class KMeans:

    def cluster(self, data, no_of_clusters, no_of_iterations=None, no_of_trials=1, normalize=False):
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
        visualize: If set to true, shows graphs of mean squared error vs. number of iterations
             as well as special graphs of the data if performing multi-class classification.
        regularization: A float that represents the regularization parameter used to prevent
            over-fitting. If set to 0, regularization is not performed.

        Returns:
        ________
        A NumPy array of NumPy arrays representing column vectors of the model parameters for
        each of the unique classifications for the data.
        """

        if data.ndim == 1:
            data = data[np.newaxis].T

        if normalize:
            data = self.normalize(data)

        if no_of_iterations == None:
            no_of_iterations = maxsize

        best_clusters = np.zeros((no_of_clusters, data.shape[1]))
        best_assignments = np.zeros(data.shape[0]).astype(int)
        best_score = maxsize

        for i in range(no_of_trials):
            clusters = data[sample(range(data.shape[0]), no_of_clusters), :]
            assignments = np.zeros(data.shape[0]).astype(int)

            previous_clusters = np.zeros(clusters.shape)
            count = 0
            keep_running = True
            while keep_running:
                for j in range(data.shape[0]):
                    min_distance = maxsize
                    cluster_index = 0
                    for k in range(no_of_clusters):
                        current_distance = self.distance(data[j, :], clusters[k, :])
                        if current_distance < min_distance:
                            min_distance = current_distance
                            cluster_index = k
                    assignments[j] = cluster_index

                for j in range(no_of_clusters):
                    current_assignments = data[assignments == j, :]
                    clusters[j, :] = np.sum(current_assignments, axis=0) / current_assignments.shape[0]

                count += 1
                keep_running = count < no_of_iterations and not np.all(clusters == previous_clusters)
                previous_clusters = clusters

            current_score = self.cost(data, clusters, assignments)
            if current_score < best_score:
                best_score = current_score
                best_clusters = clusters
                best_assignments = assignments

        return best_assignments, best_clusters

    def distance(self, x1, x2):
        return np.sum((x2 - x1) ** 2) ** 0.5

    def normalize(self, X):
        self.mean_of_features = []
        self.std_of_features = []
        temp_X = np.copy(X)
        for i in range(X.shape[1]):
            self.mean_of_features.append(np.mean(X[:, i]))
            self.std_of_features.append(np.std(X[:, i]))
            temp_X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
        return temp_X

    def cost(self, data, centroids, cluster_assignments):
        cost = 0
        for i, assignment in enumerate(cluster_assignments):
            cost += np.sum((data[i, :] - centroids[assignment, :]) ** 2)
        return cost / data.shape[0]


# Example code:

print("K-means clustering: ")
data = np.loadtxt('data/kmeans.txt', delimiter=',')

plt.scatter(data[:, 0], data[:, 1])
plt.show()

kmeans = KMeans()
assignments, clusters = kmeans.cluster(data, 3, no_of_trials=10)

plt.scatter(data[assignments == 0, 0], data[assignments == 0, 1], marker='o')
plt.scatter(data[assignments == 1, 0], data[assignments == 1, 1], marker='x')
plt.scatter(data[assignments == 2, 0], data[assignments == 2, 1], marker='1')

plt.show()