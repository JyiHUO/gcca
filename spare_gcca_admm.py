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
from numpy import linalg as LA

class spare_gcca_admm(metric):

    def __init__(self, ds, m_rank=0, mu_x = None):
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
        if mu_x == None:
            self.mu_x = [10 for i in range(len(self.list_view))]
        else:
            self.mu_x =  mu_x # [10 for i in range(len(self.list_view))]


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
            B[i] = np.linalg.pinv(sigama).dot(B[i].dot(self.G))

        return A, B


    def solve_u(self, verbose = False):
        # cal G
        self.solve_g()
        # self.G = self.G.T

        A, B = self.cal_A_B()


        for i in range(len(B)):
            U = self.linearized_bregman(A[i], B[i],self.mu_x[i], verbose=verbose)

            self.list_U.append(U)

            if verbose:
                print()
                print("next view: ")
                print()

    def solve(self):
        self.solve_u()
        self.admm()

        number_of_views = len(self.list_view)
        for i in range(number_of_views):
            projected_data = self.list_view[i].transpose().dot(self.list_U[i])

            self.list_projection.append(np.array(projected_data))

    def cal_A_B_admm(self):
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
            B[i] = np.linalg.pinv(sigama).dot(B[i])

        return A, B

    def admm(self):

        # initialize
        muta = 0.01
        beta_max = 10
        Z_new = None
        tor = 0.1
        p = 1.1

        A, B = self.cal_A_B_admm()


        for i in range(len(B)):
            tri = 0
            beta = 1 / LA.norm(A[i].T.dot(B[i]), ord=np.inf)
            while True:

                # update Z
                temp = B[i].T.dot(tri / beta - A[i].dot(self.list_U[i]) )

                U, S, V = np.linalg.svd(temp, full_matrices=False)
                Z_new = U.dot(V)

                # update G
                x = self.list_U[i] - muta*A[i].T.dot(
                    A[i].dot(self.list_U[i]) + B[i].dot(Z_new) - tri/beta
                )
                mu = muta/beta
                G_new = np.sign(x) * np.maximum(np.abs(x) - mu, 0)

                # update tri
                tri_new = tri - beta*( A[i].dot(G_new) + B[i].dot(Z_new) )

                # update beta
                beta = min(beta_max, p*beta)

                # judgement
                left = beta * LA.norm(G_new - self.list_U[i], ord="fro") / max(1, LA.norm(self.list_U[i], ord="fro") )
                right = LA.norm(tri_new - tri, ord="fro") / beta
                if left < tor and right < tor:
                    self.list_U[i] = G_new
                    tri = tri_new
                    break

                self.list_U[i] = G_new
                tri = tri_new

        self.G = Z_new


    def linearized_bregman(self, A, B, mu_x, verbose=True):
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
        # mu_x = 10
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
    data = data_generate()
    clf_ = spare_gcca_admm
#
#    # gene data
    mu_x = (20, 20)
    name = ['Srbct', 'Leukemia', 'Lymphoma', 'Prostate', 'Brain', 'Colon']

    i = 3
    data.generate_genes_data(num=i)

    print()
    print("finish reading data: ", name[i])
    print()

    # train spare gcca model
    clf = clf_(ds=data, m_rank=1, mu_x = mu_x)
    clf.solve()

    # calculate all kind of metric
    v1_test, v2_test = clf.transform(data.test_data)
    print("total correlation in training data is: ", np.sum(clf.cal_correlation(clf.list_projection)))
    print("total correlation in testing data is: ", np.sum(clf.cal_correlation([v1_test, v2_test])))
    print("training data ACC is: ", clf.cal_acc(clf.list_projection))
    print("testing data ACC is: ", clf.cal_acc([v1_test, v2_test]))
    print("each view's spare of U is ", clf.cal_spare())
    #print("total sqare is: ", clf.cal_spare()[0])

    print()
    print()