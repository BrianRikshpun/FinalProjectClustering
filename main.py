import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
import scipy.cluster.hierarchy as hc
import graphviz as gv
import re
import seaborn as sns
from clusterAnalysis import clusterAnalysis
from Models import Models

def clean_db(merged):
    '''
    :param merged: merged dataset (features + taxonomy)
    :return: cleaned dataset - ready to train
    '''

    merged = merged.drop(columns=['Unnamed: 10', 'Unnamed: 0.1', 'Unnamed: 0', 'Division', 'Assembly'])
    ranks = [f'rank {i}' for i in range(2,10)]
    merged['species'] = merged['species'].str.replace('\t', '')
    for i in ranks:
        merged[i] = merged[i].str.replace('\t', '')


    merged.pop('Organelle')
    merged.pop('Translation Table')
    merged.pop('Species')

    #merged[merged.columns[10:]]
    return merged


def plot_nas(df: pd.DataFrame):
    '''

    :param df: dataframe
    :return: Nan bar plot
    '''

    df = df.replace(r'^\s*$', np.nan, regex=True)
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %' :na_df})
        missing_data.plot(kind = "barh")
        plt.show()
    else:
        print('No NAs found')


def plotTaxonomyConnectionsAndBars(r1, r2):
    '''

    :param r1: rank 1 of the taxonomy
    :param r2: rank 2 of the taxonomy
    :return: plot
    '''

    for j in data[r1].unique():

        x = []
        y = []
        d = data.copy()
        d = d[d[r1] == j]

        for k in d[r2].unique():
            d2 = d.copy()
            d2 = d2[d2[r2] == k]
            x.append(k)
            y.append(len(d2))

        plt.bar(x,y)
        plt.title(f'Taxonomy for {j}')
        plt.show()

        # G = nx.DiGraph()
        # for index,row in d.iterrows():
        #     G.add_edge(row[r1], row[r2])
        #
        # nx.draw(G, with_labels=True)
        # plt.show()


def rankingAnalysis(data):

    for c in data.columns[8:]:
        data[c].hist()
        plt.title(f'Histogram for column {c}')
        plt.show()

    #plotTaxonomyConnectionsAndBars('rank 9', 'rank 8')


def clusterAnalysis(data, model, rank):

    clusters = model.fit_predict(data[data.columns[8:]])
    data['cluster'] = clusters
    data['rank'] = rank
    le = preprocessing.LabelEncoder()
    data['EncodedRank'] = le.fit_transform(rank)
    codonUsageClusterheatMap(data)

    for ind,i in enumerate(data['cluster'].unique()):
        x = []
        y = []
        d = data.copy()
        d = d[d['cluster'] == i]
        for j in d['EncodedRank'].unique():
            d2 = d.copy()
            d2 = d2[d2['EncodedRank'] == j]
            x.append(d2['rank'].iloc[0]) #There is only one
            y.append(len(d2))

        plt.bar(x,y)
        plt.title(f'Cluster {i} ')
        plt.show()


def codonUsageClusterheatMap(df):

    # Heatmap idx = cluster, column = codon usage bias in %

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


def addrow(df, d, name, rank, instances, ins, out):

    pass


def taxonomyDescribe(data):

    taxonomyData = pd.DataFrame(columns=['name','rank', '# instances', 'in', 'out'])
    row = {k : '' for k in taxonomyData.columns}

    for col in data.columns():
        for node in data[col].unique():
            if col == 'rank 9': #First rank
                addrow(taxonomyData, row.copy(), node, col, len(data[data[col] == node]), 0,)


if __name__ == '__main__':


    d = pd.read_csv('rankedlineage122.csv', sep = '|') #Taxnomy
    d = d.rename({d.columns[0]:'Taxid',d.columns[1]:'species', d.columns[2]:'rank 2',
                  d.columns[3]:'rank 3', d.columns[4]:'rank 4',
                  d.columns[5]:'rank 5', d.columns[6]:'rank 6',
                  d.columns[7]:'rank 7', d.columns[8]:'rank 8', d.columns[9]:'rank 9'},axis = 1)

    d2 = pd.read_csv('final_codon_dataset.csv') #Codon features
    merged = d.merge(d2, on='Taxid')
    data = clean_db(merged)

    #----- make sure the data is after clustering and with the cluster column -----
    models = Models(data[data.columns[12:]])
    models.ElbowKmeans()
    clusterAnalizer = clusterAnalysis(data)

    # rankingAnalysis(data[data.columns[:10]])
    # plot_nas(data[data.columns[:10]])
    # taxonomyDescribe(data[data.columns[:10]])
    clusterAnalysis(data[data.columns[10:]],KMeans(n_clusters=4),data['rank 9'])





