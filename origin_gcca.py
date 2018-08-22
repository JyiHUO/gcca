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
    data = data_generate()
    clf_ = gcca
#
#    # gene data
#    name = ['Srbct', 'Leukemia', 'Lymphoma', 'Prostate', 'Brain', 'Colon']
#
#    i = 3
#    data.generate_genes_data(num=i)
#
#    print()
#    print("finish reading data: ", name[i])
#    print()
#
#    # train gcca model
#    clf = clf_(ds=data, m_rank=1)
#    clf.solve()

    # calculate all kind of metric
#    v1_test, v2_test = clf.transform(data.test_data)
#    print("total correlation in training data is: ", np.sum(clf.cal_correlation(clf.list_projection)))
#    print("total correlation in testing data is: ", np.sum(clf.cal_correlation([v1_test, v2_test])))
#    print("training data ACC is: ", clf.cal_acc(clf.list_projection))
#    print("testing data ACC is: ", clf.cal_acc([v1_test, v2_test]))
#    print("each view's spare of U is ", clf.cal_spare())
#    #print("total sqare is: ", clf.cal_spare()[0])
#
#    print()
#    print()
#    clf.save_U("gcca_gene")

    # three views data for tfidf language data

    data.generate_three_view_tfidf_dataset()

    clf = clf_(ds=data, m_rank=20)
    clf.solve()

    # calculate all kind of metric
    print("reconstruction error of G in training is: ", clf.cal_G_error(data.train_data, test=False))
    print("reconstruction error of G in testing is: ", clf.cal_G_error(data.test_data, test=True))
    print("each view's spare of U is ", clf.cal_spare())
    print("total sqare is: ", np.mean(clf.cal_spare()))

    print()
    print()


#
#    # for synthetic data
#     data.generate_synthetic_dataset()
#
#     clf = clf_(ds=data, m_rank=1)
#     clf.solve()
#
#     # calculate all kind of metric
#     print("reconstruction error of G in training is: ", clf.cal_G_error(data.train_data, test=False))
#     print("reconstruction error of G in testing is: ", clf.cal_G_error(data.test_data, test=True))
#     print("each view's spare of U is ", clf.cal_spare())
#     print("total sqare is: ", np.mean(clf.cal_spare()))
#
#     print()
#     print()
#     clf.save_U("gcca_synthetic")
