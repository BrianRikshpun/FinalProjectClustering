import numpy as np
import pandas as pd
import seaborn as sns
from DistanceMatrices import DistanceMatrices
import numpy as np

DistanceMatrices = DistanceMatrices()

class EvaluationMatrices():

    def TaxonomyCloseness(self, X_train):
        node_entropy = {}
        ranks = ['rank 9', 'rank 8', 'rank 7', 'rank 6', 'rank 5', 'rank 4', 'rank 3', 'rank 2']
        for r in ranks:

            species_proba = []
            for n in X_train[r].unique():
                species_proba.append(len(X_train[X_train[r] == n]))

            species_proba = [i/len(X_train) for i in species_proba]
            node_entropy[n] = -1 * sum([i*np.log2(i) for i in species_proba])

        return sum(list(node_entropy.values()))



    def distortion(self, X_train, centers):

        return np.sum(((X_train - centers) ** 2.0).sum(axis=1))


    def silhouette(self, X_train , p):
        '''

        :param X_train: clustered data
        :param p: p of milkovski distance metric
        :return:
        '''

        #X_train includes the clusters (after fitting and predicting the clusters)

        a = []
        b = []
        s = []

        for c in X_train['cluster'].unique():

            d = X_train.copy()
            d_c = d[d['cluster'] == c]
            d_not_c = d[d['cluster'] != c]

            for index in range(len(d_c)):

                c_i = len(d_c) #Number of samples in the cluster

                if c_i == 1:
                    s.append(0)
                else:
                    p_j = d_c.iloc[[i for i in range(len(d_c)) if i != index]] #set of all points beside the i
                    p_i = pd.DataFrame(columns=p_j.columns) #set with the same len but only sample i
                    for i in range(len(d_c) - 1):
                        p_i = p_i.append(d_c.iloc[index], ignore_index = True)

                    a.append((1/(c_i - 1)) * sum(DistanceMatrices.minkowski(p_i.to_numpy(),p_j.to_numpy(), p)))

                    check_min = []
                    for not_c in d_not_c['cluster'].unique():
                        not_c_d = d_not_c.copy()
                        not_c_d = not_c_d[not_c_d['cluster'] == not_c]
                        c_not_i = len(not_c_d)

                        if len(p_i) > len(not_c_d):
                            p_i = p_i[:c_not_i]
                        if len(p_i) < len(not_c_d):
                            while(len(p_i) != len(not_c_d)):
                                p_i = p_i.append(d_c.iloc[index], ignore_index=True)
                        check_min.append((1/c_not_i) * sum(DistanceMatrices.minkowski(p_i.to_numpy(),not_c_d.to_numpy(), p)))

                    b.append(min(check_min))

        for i in range(len(a)):
            s.append((b[i] - a[i])/max(a[i],b[i]))

        return np.mean(s)


