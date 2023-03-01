import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler

from EvaluationMatrices import EvaluationMatrices
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import hamming, dice, minkowski
import numpy as np
import warnings
warnings.filterwarnings("ignore")



def makeGrid():


    #DBSCAN
    eps = [0.5, 1.0, 100.0, 1000.0]
    min_samples = [5, 100, 1000]
    metric = ['cityblock', 'cosine', 'euclidian', 'l1', 'l2', 'manhattan']

    #OPTICS
    optics_min_samples = [5,100,1000]
    optics_metric = ['cityblock', 'cosine', 'euclidian', 'l1', 'l2', 'manhattan']
    xi = [0.05, 0.5, 0.7]
    min_cluster_size = [100, 1000, 10000]
    algorithm = ['ball_tree', 'kd_tree', 'auto', 'brute']


    grid_dbscan = [(DBSCAN(eps = i, min_samples = j, metric = k, n_jobs=-1),k) for i in eps for j in min_samples for k in metric]
    grid_optics = [(OPTICS(min_samples = i, metric = j, xi = k, min_cluster_size=m, algorithm=p),j) for i in optics_min_samples for j in optics_metric
                   for k in xi for m in min_cluster_size for p in algorithm]


    return grid_dbscan, grid_optics


def findBestGrid(grid,X_train):

    global EvaluationMatrices

    X_train_df = X_train.copy()
    X_train_df = X_train_df[X_train_df.columns[12:]]
    # X_train_df = X_train_df.to_numpy()
    #X_train_df = StandardScaler().fit_transform(X_train_df)

    silhouttes = []
    taxonomyCloseness = []

    for option in grid:

        print(f"fitting model {option[0]}")
        model = option[0]
        metric = option[1]

        fitted_model = model.fit(X_train_df)
        X_train['cluster'] = fitted_model.labels_
        silhouttes.append(silhouette_score(X_train_df,fitted_model.labels_,metric=metric))
        taxonomyCloseness.append(taxonomyCloseness(X_train))

    optimal_shilhoutte = silhouttes[np.argmax(silhouttes)]
    optimal_taxonomy = taxonomyCloseness[np.argmin(taxonomyCloseness)]
    optimal_model_sh = grid[np.argmax(silhouttes)][0]
    optimal_model_tax = grid[np.argmin(taxonomyCloseness)][0]

    return optimal_shilhoutte,optimal_taxonomy,optimal_model_sh,optimal_model_tax



EvaluationMatrices = EvaluationMatrices()

data = pd.read_csv("ready_to_run.csv")

grid_dbscan, grid_optics = makeGrid()

optimal_shilhoutte_db,optimal_taxonomy_db,optimal_model_sh_db,optimal_model_tax_db = findBestGrid(grid_dbscan,data)
optimal_shilhoutte_opt,optimal_taxonomy_opt,optimal_model_sh_opt,optimal_model_tax_opt = findBestGrid(grid_optics,data)

print(f"DBSCAN - the optimal grid for db scan by shilhoutte is {optimal_model_sh_db} with score of {optimal_shilhoutte_db}")
print(f"DBSCAN - the optimal grid for db scan by taxonomyClossness is {optimal_model_tax_db} with score of {optimal_taxonomy_db}")
print(f"optics - the optimal grid for optics scan by shilhoutte is {optimal_model_sh_opt} with score of {optimal_shilhoutte_opt}")
print(f"optics - the optimal grid for optics scan by taxonomyClossness is {optimal_model_tax_opt} with score of {optimal_taxonomy_opt}")





