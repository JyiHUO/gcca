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
import time

class gcca_admm(metric):

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
            B[i] = np.linalg.pinv(sigama).dot(B[i])

        return A, B

    def solve_u(self):
        number_of_views = len(self.list_view)

        # cal G
        self.solve_g()
        print (self.G.shape)

        for i in range(number_of_views):
            U = np.linalg.pinv(self.list_view[i].transpose()) * np.mat(self.G)

            self.list_U.append(np.array(U))

    def solve(self):
        self.solve_u()
        self.admm()

        number_of_views = len(self.list_view)
        for i in range(number_of_views):
            projected_data = self.list_view[i].transpose().dot(self.list_U[i])

            self.list_projection.append(np.array(projected_data))

    def admm(self):

        # initialize
        muta = 1  # 对应论文代码的delta
        beta_max = 1e4  # 最大beta值
        Z_new = None  # 用来存放更新后的Z
        tor1 = 1e-2
        tor2 = 1e-7
        p = 2  # 对应论文代码的rho
        iter = 1e3  # 迭代次数

        A, B = self.cal_A_B()  # 对应论文的A和B


        for i in range(len(B)):  # 迭代每一个view
            N, M = self.list_U[i].shape
            iter_k = 0  # 迭代步数初始化为0
            tri = 0  # 论文倒三角符号的初始化
            G = np.random.random(size=self.list_U[i].shape)  # np.zeros_like(self.list_U[i])  # 新加坡论文G初始化为0（也就是我们的W）
            beta = 1 / np.max(np.max(np.abs(A[i].T.dot(B[i]))))  # 1 / LA.norm(A[i].T.dot(B[i]), ord=np.inf)  # beta初始化，无穷范数
            while True:
                iter_k +=1  # 迭代次数加一

                # update Z
                temp = B[i].T.dot(tri / beta - A[i].dot(G) )

                U, S, V = np.linalg.svd(temp, full_matrices=False)
                Z_new = U.dot(V.T)  # 原来是V不是V.T

                temp_Z_new = Z_new.dot(Z_new.T)  # 连这里的条件都不满足G*G.T = I, 我们的G并不一定是方阵啊，这是一个很关键的地方

                # update G
                # seperate x into piece
                # x = G - muta*A[i].T.dot(
                #     A[i].dot(G) + B[i].dot(Z_new) - tri/beta
                # )
                temp_x1 = A[i].dot(G) + B[i].dot(Z_new)
                temp_x2 = tri/beta
                temp_x3 = muta*A[i].T.dot(temp_x1 - temp_x2)
                x = G - temp_x3

                mu = muta/beta
                # seperate G_new into piece
                # G_new = np.sign(x) * np.maximum(np.abs(x) - mu, 0)
                temp_G_new1 = np.sign(x)
                temp_G_new2 = np.abs(x) - mu
                G_new = temp_G_new1 * np.maximum(temp_G_new2, 0)

                # update tri
                tri_new = tri - beta*( A[i].dot(G_new) + B[i].dot(Z_new) )

                # judgement
                left = beta * LA.norm(G_new - G, ord="fro") / max(1, LA.norm(G, ord="fro") )
                right = LA.norm(tri_new - tri, ord="fro") / beta
                if (left < tor1 and right < tor2) or iter_k == iter:  # 判断如果都小于容忍值或达到迭代次数
                    self.list_U[i] = G_new  # 把最终结果存起来
                    tri = tri_new
                    break

                # update beta
                beta = min(beta_max, p * beta)

                # 更新 k + 1的值，在这里有一点要注意，Z_{k+1}在上面已经更新了，你可以从上面代码找出来
                G = G_new
                tri = tri_new

        self.G = Z_new  # 得到最后的Z值，更新到我们论文算法的G上,有两个G,我们要用哪一个？




if __name__ == "__main__":
    data = data_generate()
    clf_ = gcca_admm


    # data.generate_synthetic_dataset()
    #
    # clf = clf_(ds=data, m_rank=1)
    # clf.solve()
    #
    # # calculate all kind of metric
    # print("reconstruction error of G in training is: ", clf.cal_G_error(data.train_data, test=False))
    # print("reconstruction error of G in testing is: ", clf.cal_G_error(data.test_data, test=True))
    # print("each view's spare of U is ", clf.cal_spare())
    # print("total sqare is: ", np.mean(clf.cal_spare()))
    #
    # print()
    # print()

    # clf.save_U("gcca_synthetic")

    # three views data for tfidf language data

    # data.generate_three_view_tfidf_dataset()
    #
    # clf = clf_(ds=data, m_rank=20)
    # clf.solve()
    #
    # # calculate all kind of metric
    # print("reconstruction error of G in training is: ", clf.cal_G_error(data.train_data, test=False))
    # print("reconstruction error of G in testing is: ", clf.cal_G_error(data.test_data, test=True))
    # print("each view's spare of U is ", clf.cal_spare())
    # print("total sqare is: ", np.mean(clf.cal_spare()))
    #
    # print()
    # print()

    # gene data

    s = time.time()

    name = ['Srbct', 'Leukemia', 'Lymphoma', 'Prostate', 'Brain', 'Colon']

    i = 0
    data.generate_genes_data(num=i)

    print()
    print("finish reading data: ", name[i])
    print()

    # train gcca model
    clf = clf_(ds=data, m_rank=1)
    clf.solve()

    # calculate all kind of metric
    v1_test, v2_test = clf.transform(data.test_data)
    print("total correlation in training data is: ", np.sum(clf.cal_correlation(clf.list_projection)))
    print("total correlation in testing data is: ", np.sum(clf.cal_correlation([v1_test, v2_test])))
    print("training data ACC is: ", clf.cal_acc(clf.list_projection))
    print("testing data ACC is: ", clf.cal_acc([v1_test, v2_test]))
    print("each view's spare of U is ", clf.cal_spare())
    # print("total sqare is: ", clf.cal_spare()[0])

    e = time.time()

    print ("total time is ", e - s)
