import numpy as np
import sklearn.datasets as ds
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from utils import *
import pandas as pd
import scipy.io as sco
import pickle

class data_generate:
    def __init__(self):
        self.views_mean = None  # list of mean
        self.views_std = None  # list of std

        self.origin_train_data = None
        self.train_data = None  # [(D, N), (D, N) ... ]
        self.test_data = None

    def generate_boston(self, normalize=True):
        boston = ds.load_boston()
        X = boston.data
        y = boston.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6,random_state=10)

        self.origin_train_data = [X_train, y_train.reshape((-1, 1))]

        if normalize:
            X_train, y_train = self._center_norm(X_train, y_train)

        self.train_data = [X_train, y_train.reshape((-1, 1))]
        self.test_data = [X_test, y_test.reshape((-1, 1))]

    def generate_mnist(self):
        data1 = load_data('noisymnist_view1.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz')
        data2 = load_data('noisymnist_view2.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz')
        x_train1 = data1[0][0]
        y_train1 = data1[0][1]
        x_test1 = data1[2][0]
        y_test1 = data1[2][1]

        x_train2 = data2[0][0]
        y_train2 = data2[0][1]
        x_test2 = data2[2][0]
        y_test2 = data2[2][1]

        self.origin_train_data = [x_train1, x_train2]


        self.train_data = [x_train1, x_train2]
        self.test_data = [x_test1, x_test2]
        self.mnist_train_label = y_train1
        self.mnist_test_label = y_test1

    def generate_mnist_x_y_acc(self):
        data1 = load_data('noisymnist_view1.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz')
        x_train1 = data1[0][0]
        y_train1 = data1[0][1]
        x_test1 = data1[2][0]
        y_test1 = data1[2][1]

        # dummy target data
        y_train = pd.get_dummies(y_train1).values
        y_test = pd.get_dummies(y_test1).values

        self.origin_train_data = [y_train1, y_test1]

        self.train_data = [x_train1, y_train]
        self.test_data = [x_test1, y_test]

    def generate_mnist_half(self):
        data1 = load_data('noisymnist_view1.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz')
        data2 = load_data('noisymnist_view2.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz')

        x_train = data1[0][0]
        y_train = data1[0][1]
        x_test = data1[2][0]
        y_test = data1[2][1]

        def _cut_half(X):
            # X = X[:6000, :]
            # X = (X - np.mean(X)) / np.std(X)

            N = X.shape[0]
            X = X.reshape((N, 28, 28))
            left = X[:, :, :14]
            right = X[:, :, 14:]
            return left.reshape((N, -1)), right.reshape((N, -1))

        self.train_data = _cut_half(x_train)
        self.test_data = _cut_half(x_test)

    def en_es_fr(self, row = 800, normalize = False):
        if row == 800:
            v1 = pd.read_csv("../gcca_data/csv_data/en800data.csv", encoding = "ISO-8859-1").values
            v2 = pd.read_csv("../gcca_data/csv_data/es800data.csv", encoding = "ISO-8859-1").values
        else:
            v1 = pd.read_csv("../gcca_data/csv_data/en1000data.csv", encoding = "ISO-8859-1").values
            v2 = pd.read_csv("../gcca_data/csv_data/fr1000data.csv", encoding = "ISO-8859-1").values

        if normalize:
            self._center_norm([v1, v2])
        v1_train, v1_test, v2_train, v2_test = train_test_split(v1, v2, test_size = 0.8, random_state = 42)

        self.train_data = [v1_train, v2_train]
        self.test_data = [v1_test, v2_test]

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

        v1_train, v1_test, v2_train, v2_test = train_test_split(v1, v2, test_size=0.6, random_state=random_state)

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
    dg.en_es_fr()