import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
from DistanceMatrices import DistanceMatrices
from EvaluationMatrices import EvaluationMatrices
import pandas as pd
import itertools
import math


DistanceMatrices = DistanceMatrices()
EvaluationMatrices = EvaluationMatrices()

class KMeans:
    def __init__(self, n_clusters=8, max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):
        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        self.centroids = [random.choice(X_train)]
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = np.sum([DistanceMatrices.euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]
        # This initial method of randomly selecting centroid starts is less effective
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]
        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = DistanceMatrices.euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = DistanceMatrices.euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs


    def elbowOptimalK(self, vals):
        x = range(1,len(vals) + 1)
        y = [float(v) for v in vals]

        TH = 0.2 #TrashHold
        ps = [(y[0] - y[1])/y[0]]
        i = 1
        while (ps[-1] >= TH) & (i <= len(vals) - 2):
            ps.append((y[i] - y[i + 1])/y[i])
            i += 1

        return i, y[i - 1]

    #
    #     # x = np.array(x)
    #     # y = np.array(y)
    #     #
    #     # z = np.polyfit(x, y, 2)
    #     # f = np.poly1d(z)
    #     #
    #     # # Get the y-values of the fitted polynomial at each x
    #     # fitted_y = f(x)
    #     # plt.plot(list(fitted_y))
    #     # plt.plot(vals)
    #     # plt.show()
    #     #
    #     # # Calculate the first derivative of the fitted polynomial
    #     # first_derivative = np.polyder(f, 1)
    #     # first_derivative_y = first_derivative(fitted_y)
    #     #
    #     # # Find the x-value where the first derivative is closest to zero
    #     # optimal_k = x[np.argmin(np.abs(first_derivative_y))]
    #     #
    #     # return int(optimal_k)
    #
    #     # dots = list(zip(vals,x))
    #     # angles = []
    #     # for i in range(len(vals) - 2):
    #     #
    #     #     a = dots[i]
    #     #     b = dots[i+1]
    #     #     c = dots[i+2]
    #     #     angles.append(math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])))
    #     #
    #     # logical_angs = [ang for ang in angles if ang <= 170]
    #     # optimal_ang = max(logical_angs)
    #     # optimal_k = angles.index(optimal_ang)
    #
    #     # diffs = []
    #     # TH = 0.
    #     # found = False
    #     # optimal_k = 0
    #     # optimal_diff = vals[0] - vals[1]
    #     #
    #     # while diffs[-1] > HT:
    #
    #
    #     return 1

# Create a dataset of 2D distributions
centers = 4
X_train, true_labels = make_blobs(n_features=2, n_samples=5000, centers=centers, random_state=42)
X_train = StandardScaler().fit_transform(X_train)
# Fit centroids to dataset

distortions_arr = []

# kmeans = KMeans(n_clusters=centers)
# kmeans.fit(X_train)
# class_centers, classification = kmeans.evaluate(X_train)
# distortions_arr.append(kmeans.distortion(X_train, class_centers))

for i in range(1,10):

    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_train)
    class_centers, classification = kmeans.evaluate(X_train)
    distortions_arr.append(EvaluationMatrices.distortion(X_train, class_centers))


optimal_k,optimal_dis = kmeans.elbowOptimalK(distortions_arr)

plt.scatter(x = optimal_k, y = optimal_dis, c = 'r', marker='+')
plt.plot(range(1,10),distortions_arr)
plt.show()


kmeans = KMeans(n_clusters=optimal_k)
kmeans.fit(X_train)
class_centers, classification = kmeans.evaluate(X_train)

# View results
class_centers, classification = kmeans.evaluate(X_train)
sns.scatterplot(x=[X[0] for X in X_train],
                y=[X[1] for X in X_train],
                hue=true_labels,
                style=classification,
                palette="deep",
                legend=None
                )
plt.plot([x for x, _ in kmeans.centroids],
         [y for _, y in kmeans.centroids],
         'k+',
         markersize=10,
         )
plt.show()



#-------------- extract the most likely k from the elbow (disortion max drop) --------
#-------------- Why the distortion is always changing!!! -----------------------------
# ------------- Walk through the algorithm implementation ----------------------------

