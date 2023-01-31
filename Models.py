from sklearn import preprocessing
from pyclustering.cluster.kmeans import kmeans,kmeans_visualizer
from pyclustering.utils.metric import distance_metric
from pyclustering.cluster.center_initializer import random_center_initializer,kmeans_plusplus_initializer
from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster.encoder import cluster_encoder
from pyclustering.cluster.elbow import elbow
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from yellowbrick.cluster import KElbowVisualizer, silhouette_visualizer

class Models:

    def __init__(self, data):
        self.data = data
        self.models = [kmeans]

    def ElbowKmeans(self):

        kmin, kmax = 2, 10
        elbow_instance = elbow(self.data, kmin, kmax)
        # process input data and obtain results of analysis
        elbow_instance.process()
        amount_clusters = elbow_instance.get_amount()  # most probable amount of clusters
        wce = elbow_instance.get_wce()  # total within-cluster errors for each K
        centers = kmeans_plusplus_initializer(self.data, amount_clusters).initialize()
        kmeans_instance = kmeans(self.data, centers)
        kmeans_instance.process()
        elbow_instance = elbow(self.data, kmin, kmax, initializer=random_center_initializer)
        elbow_instance.process()
        print(0)


    def yellowShillout(self):
        # silhouette_visualizer(KMeans(5, random_state=42), self.data , colors='yellowbrick')
        pass


