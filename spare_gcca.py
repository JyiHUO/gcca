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

class spare_gcca(metric):

    def __init__(self, ds, m_rank=0):
        '''
        Constructor for GeneralizedCCA.

        Args:
            list_view (list<ndarray>): Training data for each view
            m_rank (int): How many principal components to keep. A value of 0
                indicates that it should be full-rank. (Default 0)
        '''
        super().__init__()
        self.list_view = ds.train_data  # [(D, N), (D, N) ... ]
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

    def cal_A_B(self):
        '''
        Calculate common space of G and some necessary variable
        :param list_view: [view1, view2 ...] view shape:(D, N)
        :return: matrix G, list A , list B
        '''

        A = []
        B = []
        S = []

        for i, view in enumerate(self.list_view):
            p, s, q = np.linalg.svd(view, full_matrices=False)

            A.append(p)
            B.append(q)
            S.append(s)

            # cal A and B
            n = S[i].shape[0]
            sigama = np.zeros((n, n))
            sigama[np.arange(n), np.arange(n)] = S[i]
            A[i] = A[i].T
            B[i] = np.linalg.inv(sigama).dot(B[i].dot(self.G.T))

        return A, B




    def solve(self, verbose = False):
        # cal G
        self.solve_g()
        self.G = self.G.T

        A, B = self.cal_A_B()


        for i in range(len(B)):
            U = self.linearized_bregman(A[i], B[i], verbose=verbose)

            projected_data = self.list_view[i].transpose().dot(U)

            self.list_U.append(U)
            self.list_projection.append(projected_data)

            if verbose:
                print()
                print("next view: ")
                print()


    def linearized_bregman(self, A, B, verbose=True):
        '''
        Solve equation which is Ax = B
        :param A: matrix
        :param B: matrix
        :return: matrix X
        '''

        B = np.array(B)
        # initialize parameter
        error_x = 1
        epsilon = 1e-5
        delta = 0.5
        tau = 1
        mu_x = 5
        Numit_x = 0
        Vx_tilde = A.T.dot(B)
        Vx_old = Vx_tilde

        # solve X
        X = None
        while error_x > epsilon:
            # print (Vx_tilde)
            t = delta * np.sign(Vx_tilde)
            b = np.maximum(tau * np.abs(Vx_tilde) - mu_x, 0)
            X = t * b
            # X = delta * np.sign(Vx_tilde) * np.maximum(tau * np.abs(Vx_tilde) - mu_x, 0)
            # print(Swx.shape)
            Vx_new = Vx_tilde - A.T.dot(A.dot(X) - B)
            alpha = (2 * Numit_x + 3) / (Numit_x + 3)
            Vx_tilde = alpha * Vx_new + (1 - alpha) * Vx_old

            Vx_old = Vx_new

            error_x = np.linalg.norm(A.dot(X) - B, "fro") / np.linalg.norm(B, "fro")
            Numit_x = Numit_x + 1

            if verbose:
                if Numit_x % 200 == 0:
                    print(error_x)

        return X




if __name__ == "__main__":
    # read some data
    # generate boston data
    data = data_generate()
    # dg.generate_boston()
    # dg.generate_mnist(normalize=False)
    # dg.generate_mnist_half()
    # dg.en_es_fr(800)

    for i in range(6):
        data.generate_genes_data(num=i)
        # dg.generate_twitter_dataset()

        print()
        print("finish reading data")
        print()

        # train gcca model
        gcca = spare_gcca(ds=data, m_rank=2)
        gcca.solve(verbose=False)

        # calculate all kind of metric
        print("total correlation is: ", np.mean(gcca.cal_correlation()))
        print("training data ACC is: ", gcca.cal_acc(gcca.list_projection))
        print("testing data ACC is: ", gcca.cal_acc(gcca.transform(data.test_data[0].T, data.test_data[1].T)))
        print("each view's spare of U is ", gcca.cal_spare())
        print("total sqare is: ", gcca.cal_spare()[0])

        print ()
        print ()