import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
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
    #metric = ['kulsinski', 'canberra', 'rogerstanimoto', 'matching', 'cityblock',  'chebyshev', 'cosine', 'manhattan', 'correlation', 'jaccard', 'braycurtis',  'sokalmichener', 'nan_euclidean', 'sokalsneath',
              #'hamming',  'euclidean', 'dice', 'l2', 'l1', 'yule', 'sqeuclidean',  'russellrao']

    metric = ['hamming', 'euclidean', 'l2', 'l1', 'yule']

    # eps = [0.5, 1.0]
    # min_samples = [5]
    # metric = ['cityblock', 'cosine']

    #OPTICS
    optics_min_samples = [5,100,1000]
    optics_metric = ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']
    xi = [0.05, 0.5, 0.7]
    min_cluster_size = [100, 1000, 10000]
    algorithm = ['ball_tree', 'kd_tree', 'auto', 'brute']

    # optics_min_samples = [5,100,1000]
    # optics_metric = ['cityblock']
    # xi = [0.05]
    # min_cluster_size = [100]
    # algorithm = ['ball_tree', 'kd_tree']


    grid_dbscan = [(DBSCAN(eps = i, min_samples = j, metric = k, n_jobs=-1),k) for i in eps for j in min_samples for k in metric]
    grid_optics = [(OPTICS(min_samples = i, metric = j, xi = k, min_cluster_size=m, algorithm=p , n_jobs=-1),j) for i in optics_min_samples for j in optics_metric
                   for k in xi for m in min_cluster_size for p in algorithm]


    return grid_dbscan, grid_optics


def findBestGrid(grid,X_train):


    X_train_df = X_train.copy()
    X_train_df = X_train_df[X_train_df.columns[12:]]
    # X_train_df = X_train_df.to_numpy()
    #X_train_df = StandardScaler().fit_transform(X_train_df)

    silhouttes = []
    taxonomyCloseness = []
    k = []

    grid = grid[1:]

    for option in grid:

        print(f"fitting model {option[0]}")
        model = option[0]
        metric = option[1]

        fitted_model = model.fit(X_train_df)
        print("model fitted")
        X_train['cluster'] = fitted_model.labels_
        k.append(f"k = {len(X_train['cluster'].unique())}")
        print("Starting silhoutte")
        silhouttes.append(silhouette_score(X_train_df,fitted_model.labels_,metric=metric))
        #silhouttes.append(0)
        print("starting taxonomy closeness")
        taxonomyCloseness.append(evaluationMatrices.TaxonomyCloseness(X_train))
        print(f"for the fitted model, the taxonomy clossness is {taxonomyCloseness[-1]}, silhoutte is {silhouttes[-1]} and the k = {k[-1]}")

    optimal_shilhoutte = silhouttes[np.argmax(silhouttes)]
    optimal_taxonomy = taxonomyCloseness[np.argmin(taxonomyCloseness)]
    optimal_model_sh = grid[np.argmax(silhouttes)][0]
    optimal_model_tax = grid[np.argmin(taxonomyCloseness)][0]

    return optimal_shilhoutte,optimal_taxonomy,optimal_model_sh,optimal_model_tax




def ClusterEachTaxonomy(df, rank):

    for cluster in df['clusters'].unique():
        x = []
        y = []
        d = df.copy()
        d = d[d['clusters'] == cluster]

        for tax in d[rank].unique():
            x.append(tax)
            y.append(len(d[d[rank] == tax]))

        plt.bar(x,y)
        plt.title(f"histogram of taxonomy in cluster {cluster} for {rank}")
        plt.xlabel("taxonomy")
        plt.ylabel("count")
        plt.savefig(f"CET {cluster} {rank}.jpg")
        plt.show()

def TaxonomyEachCluster(df , rank):

    for tax in df[rank].unique():
        x = []
        y = []
        d = df.copy()
        d = d[d[rank] == tax]

        for cluster in d['clusters'].unique():
            x.append(cluster)
            y.append(len(d[d['clusters'] == cluster]))

        plt.bar(x, y)
        plt.title(f"histogram of clusters in taxonomy {tax} for {rank}")
        plt.xlabel("cluster")
        plt.ylabel("count")
        plt.savefig(f"TEC {tax} {rank}.jpg")
        plt.show()



def doKMeans(X_train, max_k):


    X_train_df = X_train.copy()
    X_train_df = X_train_df[X_train_df.columns[14:-1]]
    TC = []


    for i in range(2,max_k):

        print(f'fitting clusters = {i}')
        kmeans = KMeans(n_clusters=i).fit(X_train_df)
        #X_train = X_train[X_train.columns[1:11]]
        X_train['clusters'] = kmeans.labels_
        TC.append(evaluationMatrices.calcTreeTaxonomy(X_train[list(X_train.columns[2:11]) + ['clusters']]))
        print(f'score is : {TC[-1]}')

    print(f'min arg is {np.argmin(TC)}')
    print(TC)
    kmeans = KMeans(n_clusters=(np.argmin(TC) + 2 )).fit(X_train_df)
    X_train_df['clusters'] = kmeans.labels_
    ClusterEachTaxonomy(X_train_df, 'rank_9')
    TaxonomyEachCluster(X_train_df, 'rank_9')
    X_train_df.to_csv("first optimal split.csv")



evaluationMatrices = EvaluationMatrices()

data = pd.read_csv("ready_to_run.csv")

# data1 = pd.read_csv("cluster0.csv")
# data2 = pd.read_csv("cluster1.csv")
#
# data1 = data1.rename(columns={'clusters':'clustered_rank_9'})
# data2 = data2.rename(columns={'clusters':'clustered_rank_9'})

# data1 = data1[:10000]
# data2 = data2[:10000]
# doKMeans(data1,20)
# doKMeans(data2,20)

doKMeans(data,8)



# grid_dbscan, grid_optics = makeGrid()
#
# optimal_shilhoutte_opt,optimal_taxonomy_opt,optimal_model_sh_opt,optimal_model_tax_opt = findBestGrid(grid_optics,data)
# optimal_shilhoutte_db,optimal_taxonomy_db,optimal_model_sh_db,optimal_model_tax_db = findBestGrid(grid_dbscan,data)
#
# print(f"DBSCAN - the optimal grid for db scan by shilhoutte is {optimal_model_sh_db} with score of {optimal_shilhoutte_db}")
# print(f"DBSCAN - the optimal grid for db scan by taxonomyClossness is {optimal_model_tax_db} with score of {optimal_taxonomy_db}")
# print(f"optics - the optimal grid for optics scan by shilhoutte is {optimal_model_sh_opt} with score of {optimal_shilhoutte_opt}")
# print(f"optics - the optimal grid for optics scan by taxonomyClossness is {optimal_model_tax_opt} with score of {optimal_taxonomy_opt}")






