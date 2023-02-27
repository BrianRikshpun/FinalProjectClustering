from sklearn.cluster import DBSCAN, OPTICS
from EvaluationMatrices import EvaluationMatrices
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import hamming, dice, minkowski



def makeGrid():


    #DBSCAN
    eps = [0.5, 1.0, 100.0, 1000.0]
    min_samples = [5, 100, 1000]
    metric = ['cityblock', 'cosine', 'euclidian', 'l1', 'l2', 'manhattan', hamming()]

    #OPTICS
    optics_min_samples = [5,100,1000]
    optics_metric = ['cityblock', 'cosine', 'euclidian', 'l1', 'l2', 'manhattan', hamming()]
    xi = [0.05, 0.5, 0.7]
    min_cluster_size = [100, 1000, 10000]
    algorithm = ['ball_tree', 'kd_tree', 'auto', 'brute']


    grid_dbscan = [(DBSCAN(eps = i, min_samples = j, metric = k, n_jobs=-1),k) for i in eps for j in min_samples for k in metric]
    grid_optics = [(OPTICS(min_samples = i, metric = j, xi = k, min_cluster_size=m, algorithm=p),j) for i in optics_min_samples for j in optics_metric
                   for k in xi for m in min_cluster_size for p in algorithm]


    return grid_dbscan, grid_optics


def findBestGrid(grid,X):

    silhouttes = []
    taxonomyCloseness = []

    for option in grid:

        model = option[0]
        metric = option[1]

        fitted_model = model.fit(X)
        X['cluster'] = fitted_model.labels_
        silhouttes.append(silhouette_score(X,fitted_model.labels_,metric=metric))




EvaluationMatrices = EvaluationMatrices()



