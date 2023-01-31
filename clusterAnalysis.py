import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns

class clusterAnalysis:
    def __init__(self, data):
        self.data = data


    def codonUsageClusterheatMap(df):

        # Heatmap idx = cluster, column = codon usage bias in %

        # clusters = model.fit_predict(df[df.columns[8:]])
        # df['cluster'] = clusters

        cols = list(df.columns)[:-3]
        idx = list(df['cluster'].unique())
        add_row = {}
        heat_df = pd.DataFrame(columns=cols)

        for i in idx:
            d = df.copy()
            for col in cols:
                add_row[col] = np.average(d[d['cluster'] == i][col])
            heat_df = heat_df.append(add_row, ignore_index=True)

        heat_df.style.background_gradient(cmap='viridis') \
            .set_properties(**{'font-size': '20px'})

        sns.heatmap(heat_df[heat_df.columns[7:-3]])
        plt.show()


    def clusterAnalysis(data, model, rank):

        clusters = model.fit_predict(data[data.columns[8:]])
        data['cluster'] = clusters
        data['rank'] = rank
        le = preprocessing.LabelEncoder()
        data['EncodedRank'] = le.fit_transform(rank)

        for ind, i in enumerate(data['cluster'].unique()):
            x = []
            y = []
            d = data.copy()
            d = d[d['cluster'] == i]
            for j in d['EncodedRank'].unique():
                d2 = d.copy()
                d2 = d2[d2['EncodedRank'] == j]
                x.append(d2['rank'].iloc[0])  # There is only one
                y.append(len(d2))

            plt.bar(x, y)
            plt.title(f'Cluster {i} ')
            plt.show()