import numpy as np
import sklearn.datasets as ds
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from utils import *
import pandas as pd
import scipy.io as sco
import pickle
from data_class import *
from metric import *

class gcca(metric):

    def __init__(self, ds, m_rank=0):
        '''
        Constructor for GeneralizedCCA.

        Args:
            list_view (list<ndarray>): Training data for each view
            m_rank (int): How many principal components to keep. A value of 0
                indicates that it should be full-rank. (Default 0)
        '''

        super().__init__()
        self.list_view = [dd.T for dd in ds.train_data]  # [(D, N), (D, N) ... ]
        self.ds = ds
        self.m_rank = m_rank  # top_r
        self.G = None  # subspace
        self.list_U = []  # save U for each view [(D, r), (D, r) ... ]
        self.list_projection = []  # save project data through U for each view [(N, r), (N, r) ... ]


    def solve_g(self):
        '''
        Solves MAX-VAR GCCA optimization problem and returns the matrix G

        Returns:
            numpy.ndarray, the matrix 'G' that solves GCCA optimization problem
        '''

        reg = 0.00000001  # regularization parameter

        M = []  # matrix corresponding to M^tilde

        for i in range(len(self.list_view)):
            X = self.list_view[i].transpose()  # (N, D) (100, 17)

            # Perform rank-m SVD of X_j which yields X_j = A_j*S_j*B_j^T
            A, S, B = np.linalg.svd(X, full_matrices=False)
            # A:(N, m) (100, 17)
            # S:(17,)
            # B:(m, D) (17, 17)

            S = np.diag(S)

            N = np.shape(A)[0]
            m = np.shape(S)[0]

            # Compute and store A_J*T_J where T_j*T_j^T = S_j^T(r_jI+S_jS_j^T)^(-1)S_j
            # T = np.sqrt(np.mat(S.transpose()) * np.linalg.inv(reg * np.identity(m) + np.mat(S) * np.mat(S.transpose())) * np.mat(S))
            # (17, 17) diagonal matrix

            # Create an N by mJ matrix 'M^tilde' which is given by [A_1*T_1 ... A_J*T_J]
            if i == 0:
                M = np.array([], dtype=np.double).reshape(N, 0)

            # Append to existing M^tilde
            # M = np.hstack((M, np.mat(A) * np.mat(T)))  # (100, 54) (N, D1 + D2 + D3)
            M = np.hstack((M, np.mat(A)))

        # Perform SVD on M^tilde which yields G*S*V^T
        G, S, V = np.linalg.svd(M, full_matrices=False)
        # G (100, 54) (N, D_all)
        # S (54)
        # V (54, 54)

        if self.m_rank != 0:
            G = G[:, 0:self.m_rank]

        # Finally, return matrix G which has been computed from above

        self.G = G

        # return G  # (N, D_all or r)



    def solve(self):
        number_of_views = len(self.list_view)

        # cal G
        self.solve_g()
        # print (self.G.shape)

        for i in range(number_of_views):
            U = np.linalg.pinv(self.list_view[i].transpose()) * np.mat(self.G)

            projected_data = np.mat(self.list_view[i].transpose()) * np.mat(U)

            self.list_U.append(np.array(U))
            self.list_projection.append(np.array(projected_data))





if __name__ == "__main__":

    # read some data
    # generate boston data
    data = data_generate()
    # dg.generate_boston()
    # dg.generate_mnist(normalize=False)
    # dg.generate_mnist_half()
    # dg.en_es_fr(800)
    # data.generate_genes_data(num=0)
    data.generate_twitter_dataset()

    print()
    print("finish reading data")
    print()

    # train gcca model
    clf = gcca(ds=data, m_rank=250)
    clf.solve()

    # calculate all kind of metric
    print("total correlation is: ", np.mean(clf.cal_correlation()))
    # print("training data ACC is: ", clf.cal_acc(clf.list_projection))
    # print("testing data ACC is: ", clf.cal_acc(clf.transform(data.test_data[0].T, data.test_data[1].T)))
    print("each view's spare of U is ", clf.cal_spare())
    print("total sqare is: ", clf.cal_spare()[0])
