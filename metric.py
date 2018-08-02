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


class metric:
    def __init__(self):
        self.list_projection = []
        self.list_U = []
        self.ds = None

    def cal_correlation(self):
        # print (123)
        def rank_corr(corr_array):
            D = int(corr_array.shape[0] / 2)
            res = []
            for i in range(D):
                res.append(corr_array[i][i + D])
            return res

        corr_array = np.corrcoef(np.concatenate(self.list_projection, axis=1), rowvar=False)
        return rank_corr(corr_array)

    def cal_r2_score(self):
        return r2_score(self.list_projection[0], self.list_projection[1]), r2_score(self.list_projection[1], self.list_projection[0])

    def cal_spare(self):
        res = []
        for u in self.list_U:
            res.append(np.sum(u == 0) / (u.shape[0] * u.shape[1]))
        return res

    def cal_average_precision(self, list_projection):
        '''
        list_projection: [(N, D), (N, D) ... ]
        '''

        v1 = list_projection[0]
        v2 = list_projection[1]

        N = v1.shape[0]

        precision = 0
        for i in range(N):
            temp = []
            for j in range(N):
                dist = np.sum((v1[i] - v2[j]) ** 2)
                temp.append((dist, j))
            temp = sorted(temp, key=lambda x: x[0], reverse=True)  # least distance is the best

            index = None
            for it, t in enumerate(temp):
                if t[1] == i:
                    index = it
                    break
            # print (index)
            precision += float(index + 1) / N
        precision /= N
        return precision

    def cal_acc(self, list_projection):
        v1 = list_projection[0]
        v2 = list_projection[1]

        N = v1.shape[0]

        precision = 0

        # some pre for y
        label = set()
        for arr in v2:
            label.add(tuple(arr))

        label = list(label)
        res = []
        for arr in v2:
            for i, t in enumerate(label):
                if tuple(arr) == t:
                    res.append(i)
                    break
        c = 0
        for i in range(N):
            temp = []
            for j in range(N):
                dist = np.sum((v1[i] - v2[j]) ** 2)
                temp.append((dist, j))
            temp = sorted(temp, key=lambda x: x[0])
            for iz, z in enumerate(label):
                tt = tuple(v2[temp[0][1]])
                if tt == z:
                    if iz == res[i]:
                        c += 1
        return float(c) / N

    def transform(self, v1, v2):
        '''
        :param v1: (N, D)
        :param v2:
        :return:
        '''
        U1, U2 = self.list_U
        return v1.dot(U1), v2.dot(U2)

    def predict(self, X):
        '''
        X: (N, D)
        '''
        X = X.copy()
        X -= self.ds.x_mean
        X /= self.ds.x_std
        X_proj = X.dot(self.list_U[0])
        y_pred = X_proj.dot(np.linalg.pinv(self.list_U[1])) * self.ds.y_std + self.ds.y_mean
        return y_pred