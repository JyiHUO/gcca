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

        big_A_D1 = 0
        big_A_D2 = 0
        for i in range(len(self.list_view)):
            DA1, DA2 = A[i].shape
            big_A_D1 += DA1
            big_A_D2 += DA2

        big_A = np.zeros(shape=(big_A_D1, big_A_D2))
        big_B = np.concatenate(B, axis=0)

        row_index_A = 0
        col_index_A = 0
        for i in range(len(self.list_view)):
            DA1, DA2 = A[i].shape
            big_A[row_index_A:row_index_A + DA1, col_index_A:col_index_A + DA2] = A[i]

            row_index_A += DA1
            col_index_A += DA2

        return big_A, big_B


    def solve(self):
        # self.solve_u()
        self.admm()

    def admm(self):

        # initialize
        muta = 1  # 对应论文代码的delta
        beta_max = 1e4  # 最大beta值
        Z_new = None  # 用来存放更新后的Z
        tor1 = 1e-5
        tor2 = 1e-7
        p = 2  # 对应论文代码的rho
        iter = 1000 # 迭代次数

        A, B = self.cal_A_B()  # 对应论文的A和B


        iter_k = 0  # 迭代步数初始化为0
        tri = 0  # 论文倒三角符号的初始化
        G = np.zeros(shape=(A.shape[1], self.m_rank))      # 新加坡论文G初始化为0（也就是我们的W）
        beta = 1 / np.max(np.max(np.abs(A.T.dot(B))))  # 1 / LA.norm(A[i].T.dot(B[i]), ord=np.inf)  # beta初始化，无穷范数
        while True:
            iter_k +=1  # 迭代次数加一

            # update Z
            temp = B.T.dot(tri / beta - A.dot(G) )

            U1, S, V = np.linalg.svd(temp, full_matrices=False)
            Z_new = U1.dot(V.T)  # 原来是V不是V.T

            temp_Z_new = Z_new.dot(Z_new.T)  # 连这里的条件都不满足G*G.T = I, 我们的G并不一定是方阵啊，这是一个很关键的地方

            # update G
            # seperate x into piece
            # x = G - muta*A[i].T.dot(
            #     A[i].dot(G) + B[i].dot(Z_new) - tri/beta
            # )
            temp_x1 = A.dot(G) + B.dot(Z_new)
            temp_x2 = tri/beta
            temp_x3 = muta*A.T.dot(temp_x1 - temp_x2)
            x = G - temp_x3

            mu = muta/beta
            # seperate G_new into piece
            # G_new = np.sign(x) * np.maximum(np.abs(x) - mu, 0)
            temp_G_new1 = np.sign(x)
            temp_G_new2 = np.abs(x) - mu
            G_new = temp_G_new1 * np.maximum(temp_G_new2, 0)

            # update tri
            tri_new = tri - beta*( A.dot(G_new) + B.dot(Z_new) )

            # judgement
            left = beta * LA.norm(G_new - G, ord="fro") / max(1, LA.norm(G, ord="fro") )
            right = LA.norm(tri_new - tri, ord="fro") / beta
            if (left < tor1 and right < tor2) or iter_k == iter:  # 判断如果都小于容忍值或达到迭代次数
                G = G_new  # 把最终结果存起来
                tri = tri_new
                break

            # update beta
            beta = min(beta_max, p * beta)

            # 更新 k + 1的值，在这里有一点要注意，Z_{k+1}在上面已经更新了，你可以从上面代码找出来
            G = G_new
            tri = tri_new

        self.G = Z_new  # 得到最后的Z值，更新到我们论文算法的G上,有两个G,我们要用哪一个？
        selected_index = 0
        for i in range(len(self.list_view)):  # (D,N)
            selected_d = self.list_view[i].shape[0]
            U_selected = G[selected_index:selected_index + selected_d, :]
            projected_data = self.list_view[i].transpose().dot(U_selected)
            self.list_U.append(U_selected)
            self.list_projection.append(projected_data)
            selected_index += selected_d




if __name__ == "__main__":
    data = data_generate()
    clf_ = gcca_admm

    # --------------------------------------------
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

    # s = time.time()
    #
    # name = ['Srbct', 'Leukemia', 'Lymphoma', 'Prostate', 'Brain', 'Colon']
    #
    # i = 5
    # data.generate_genes_data(num=i)
    #
    # print()
    # print("finish reading data: ", name[i])
    # print()
    #
    # # train gcca model
    # clf = clf_(ds=data, m_rank=1)
    # clf.solve()
    #
    # # calculate all kind of metric
    # v1_test, v2_test = clf.transform(data.test_data)
    # print("total correlation in training data is: ", np.sum(clf.cal_correlation(clf.list_projection)))
    # print("total correlation in testing data is: ", np.sum(clf.cal_correlation([v1_test, v2_test])))
    # print("training data ACC is: ", clf.cal_acc(clf.list_projection))
    # print("testing data ACC is: ", clf.cal_acc([v1_test, v2_test]))
    # print("each view's spare of U is ", clf.cal_spare())
    # # print("total sqare is: ", clf.cal_spare()[0])
    #
    # e = time.time()
    #
    # print ("total time is ", e - s)


    # multi view lang
    data.generate_multi_view_tfidf_dataset()

    clf = clf_(ds=data, m_rank=20)
    clf.solve()

    # calculate all kind of metric
    print("reconstruction error of G in training is: ", clf.cal_G_error(data.train_data, test=False))
    print("reconstruction error of G in testing is: ", clf.cal_G_error(data.test_data, test=True))
    print("each view's spare of U is ", clf.cal_spare())
    print("total sqare is: ", np.mean(clf.cal_spare()))

    print()
    print()
