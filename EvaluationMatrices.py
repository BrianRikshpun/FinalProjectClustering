import numpy as np

class EvaluationMatrices():

    def distortion(self, X_train, centers):

        return np.sum(((X_train - centers) ** 2.0).sum(axis=1))





