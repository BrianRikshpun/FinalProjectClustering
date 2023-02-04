import numpy as np
from decimal import Decimal


class DistanceMatrices():


    def euclidean(self,point, data):
        """
        Euclidean distance between point & data.
        Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
        """
        return np.sqrt(np.sum((point - data)**2, axis=1))

    def manhattan(self, point, data):
        """
        Return the manhattan distance between points p and q
        assuming both to have the same number of dimensions
        """
        # sum of absolute difference between coordinates

        return np.sum(np.abs(point - data), axis = 1)


    def minkowski(self, point, data, p):
        """
        Return the minkowski distance between points p and q
        assuming both to have the same number of dimensions
        """

        return np.power(np.sum(np.power(np.abs(np.abs(point - data)),p), axis = 1), 1/p)

