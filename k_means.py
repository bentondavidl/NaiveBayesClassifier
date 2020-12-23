import pandas as pd
import numpy as np


class k_means:
    """Implementation of the KMeans algorithm using Euclidean distance

    Usage:
    ```Python
    cluster = k_means(3)
    cluster.fit(data)
    print(cluster.centroids)
    print(cluster.classes)
    """

    def __init__(self, k=8, max_iter=100):
        """Initialize KMeans algorithm

        Parameters
        ----------
        k : int, optional
            Number of clusters to create, by default 8
        max_iter : int, optional
            Maximum number of iterations to wait for stabilization, by default 100
        """
        self.k = k
        self.max_iter = max_iter

    def euclid_distance(self, a, b):
        """Calculate the distance between two vectors using the Euclidean method

        Parameters
        ----------
        a : np.array
            first vector (usually centroid)
        b : np.array
            second vector

        Returns
        -------
        np.array
            A vector that represents the distance between the two points
        """
        return np.sqrt(np.sum(np.square(a - b), axis=1))

    def fit(self, X: pd.DataFrame):
        """Find centroids based on the provided data and group the data into classes

        Parameters
        ----------
        X : pd.DataFrame
            Data to be clustered
        """
        self.centroids = X.iloc[:self.k].to_numpy()

        for x in range(self.max_iter):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []
                for j in range(len(X)):
                    distances = self.euclid_distance(
                        self.centroids[i], X.iloc[j].to_numpy())
                    classification = np.argmin(distances)
                    self.classes[classification].append(X.iloc[j])

            previous = np.array(self.centroids)

            for class_ in self.classes:
                self.centroids[class_] = np.average(
                    self.classes[class_], axis=0)

            curr = self.centroids

            if not np.sum((curr - previous)/previous * 100) > 0.0001:
                break
