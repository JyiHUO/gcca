import numpy as np
import sklearn.datasets as ds
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from utils import *
import pandas as pd
import scipy.io as sco
import pickle
from sklearn.decomposition import TruncatedSVD as PCA

class data_generate:
    def __init__(self):
        self.views_mean = None  # list of mean
        self.views_std = None  # list of std

        self.origin_train_data = None
        self.train_data = None  # [(D, N), (D, N) ... ]
        self.test_data = None


    # def en_es_fr(self, row = 800, normalize = False):
    #     if row == 800:
    #         v1 = pd.read_csv("../gcca_data/csv_data/en800data.csv", encoding = "ISO-8859-1").values
    #         v2 = pd.read_csv("../gcca_data/csv_data/es800data.csv", encoding = "ISO-8859-1").values
    #     else:
    #         v1 = pd.read_csv("../gcca_data/csv_data/en1000data.csv", encoding = "ISO-8859-1").values
    #         v2 = pd.read_csv("../gcca_data/csv_data/fr1000data.csv", encoding = "ISO-8859-1").values
    #
    #     if normalize:
    #         self._center_norm([v1, v2])
    #     v1_train, v1_test, v2_train, v2_test = train_test_split(v1, v2, test_size = 0.8, random_state = 42)
    #
    #     self.train_data = [v1_train, v2_train]
    #     self.test_data = [v1_test, v2_test]

    def generate_genes_data(self,num=0, normalize=False, random_state = 42):
        Srbct = sco.loadmat("../gcca_data/genes_data/Srbct.mat")
        Leukemia = sco.loadmat("../gcca_data/genes_data/Leukemia.mat")
        Lymphoma = sco.loadmat("../gcca_data/genes_data/Lymphoma.mat")
        Prostate = sco.loadmat("../gcca_data/genes_data/Prostate.mat")
        Brain = sco.loadmat("../gcca_data/genes_data/Brain.mat")
        Colon = sco.loadmat("../gcca_data/genes_data/Colon.mat")

        name = ['Srbct', 'Leukemia', 'Lymphoma', 'Prostate', 'Brain', 'Colon']
        # print(name[num])
        data = [Srbct, Leukemia, Lymphoma, Prostate, Brain, Colon]

        v1 = data[num]["fea"]
        v2 = pd.get_dummies(data[num]["gnd"].reshape((-1))).values


        self.origin_train_data = [v1, data[num]["gnd"].reshape((-1))]
        if normalize:
            # v1, v2 = self._center_norm([v1, v2])
            v1 = data[num]["fea"]
            v2 = data[num]["gnd"]
            v1, v2 = self._center_norm([v1, v2])
            v2 = pd.get_dummies(v2.reshape((-1))).values

        v1_train, v1_test, v2_train, v2_test = train_test_split(v1, v2, test_size=0.5, random_state=random_state)

        # print ("- the shape of training data in view one is: ", v1_train.shape)
        # print ("- the shape of training data in view two is: ", v2_train.shape)
        # print("- the shape of testing data in view one is: ", v1_test.shape)
        # print("- the shape of testing data in view two is: ", v1_test.shape)

        self.train_data = [v1_train, v2_train]
        self.test_data = [v1_test, v2_test]

    def generate_twitter_dataset(self, views_to_keep=4, normalize=True):
        with open("../gcca_data/twitter/views.pickle", 'rb') as f:
            views = pickle.load(f)  # 6 views. the one of them is [102380, 1000]
        with open("../gcca_data/twitter/train_label.pickle", 'rb') as f:
            train_label = pickle.load(f)  # [102380, 250]
        with open("../gcca_data/twitter/test_label.pickle", 'rb') as f:
            test_label = pickle.load(f)  # [102380, 250]

        if views_to_keep == 4:
            views.pop()
            views.pop()


        if normalize:
            views = self._center_norm(views)

        self.train_data = [v[:, :] for v in views]
        self.train_data.append(train_label[:, :])
        self.test_data = [v[:, :] for v in views]
        self.test_data.append(test_label[:, :])

    def generate_three_view_tfidf_dataset(self):
        with open("../gcca_data/three_view_data/big_dict.pickle", 'rb') as f:
            data_dict = pickle.load(f)

        val = list(data_dict.values())


        index = np.arange(val[0].shape[0])
        index_train, index_test = train_test_split(index, test_size=0.5, random_state=42)

        self.train_data = [val[i][index_train, :] for i in range(3)]
        self.test_data = [val[i][index_test, :] for i in range(3)]

    def generate_synthetic_dataset(self):
        v1 = np.concatenate([np.ones(1000), np.ones(2000)*-1, np.zeros(7000)]).reshape((-1, 1))
        v2 = np.concatenate([np.zeros(12000), np.ones(1000), np.ones(2000)*-1]).reshape((-1, 1))
        v3 = np.concatenate([np.ones(2000), np.zeros(15000), np.ones(1000)*-1]).reshape((-1, 1))

        mu1 = np.random.normal(loc=0, scale=0.3, size=(10000, 100))
        mu2 = np.random.normal(loc=0, scale=0.4, size=(15000, 100))
        mu3 = np.random.normal(loc=0, scale=0.5, size=(18000, 100))

        u = np.random.normal(loc=0, scale=1, size=(1, 100))

        index = np.arange(100)
        index_train, index_test = train_test_split(index, test_size=0.5, random_state=42)

        d1 = v1.dot(u) + mu1  # (D, N)
        d2 = v2.dot(u) + mu2
        d3 = v3.dot(u) + mu3

        self.train_data = [d1.T[index_train], d2.T[index_train], d3.T[index_train]]
        self.test_data = [d1.T[index_test], d2.T[index_test], d3.T[index_test]]

    def _center_norm(self, views):

        N = views[0].shape[0]
        for i in range(len(views)):
            v_mean = views[i].mean(0)
            v_std = views[i].std(0)
            views[i] = views[i].reshape((N, -1))
            views[i] = (views[i] - v_mean) / v_std

        return views

if __name__ == "__main__":
    dg = data_generate()
    dg.generate_three_view_tfidf_dataset()
    print([dg.train_data[i].shape for i in range(len(dg.train_data))])
    print([dg.test_data[i].shape for i in range(len(dg.test_data))])