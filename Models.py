from sklearn import preprocessing
from pyclustering.cluster.kmeans import kmeans,kmeans_visualizer
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils.metric import distance_metric
from pyclustering.cluster.center_initializer import random_center_initializer,kmeans_plusplus_initializer
from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster.encoder import cluster_encoder
from pyclustering.cluster.elbow import elbow
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from matplotlib import pyplot as plt

class Models:

    def __init__(self, data):
        self.data = data
        self.models = [kmeans]

    def ElbowKmeans(self):

        kmin, kmax = 1, 10
        elbow_instance = elbow(self.data.values.tolist(), kmin, kmax, initializer=random_center_initializer)
        elbow_instance.process()
        amount_clusters = elbow_instance.get_amount()
        wce = elbow_instance.get_wce()  # total within-cluster errors for each K

        figure = plt.figure(1)
        ax = figure.add_subplot(111)
        ax.plot(range(kmin, kmax + 1), wce, color='b', marker='.')
        ax.plot(amount_clusters, wce[amount_clusters - kmin], color='r', marker='.', markersize=10)
        ax.annotate("Elbow", (amount_clusters + 0.1, wce[amount_clusters - kmin] + 5))
        ax.grid(True)
        plt.ylabel("WCE")
        plt.xlabel("K")
        plt.show()
        print(0)


    def yellowShillout(self):
        # silhouette_visualizer(KMeans(5, random_state=42), self.data , colors='yellowbrick')
        pass


