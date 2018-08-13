from sklearn.cross_decomposition import CCA
import numpy as np
import pandas as pd
from metric import *
from data_class import *

class cca(metric):
    def __init__(self, ds, m_rank):
        super().__init__()
        self.list_view = [dd.T for dd in ds.train_data]  # [(D, N), (D, N) ... ]
        self.ds = ds
        self.list_U = []  # save U for each view [(D, r), (D, r) ... ]
        self.list_projection = []  # save project data through U for each view [(N, r), (N, r) ... ]
        self.m_rank = m_rank

        self.model = None

    def solve(self):
        v1, v2 = self.list_view
        clf = CCA(n_components=self.m_rank)
        clf = clf.fit(v1.T, v2.T)
        self.model = clf

        X_c, Y_c = clf.transform(v1.T, v2.T)
        self.list_projection = [X_c, Y_c]
        self.list_U = [clf.x_rotations_, clf.y_rotations_]

    def transform(self, list_view):
        '''
        :param v1: (N, D)
        :param v2:
        :return:
        '''
        v1, v2 = list_view
        X_c, Y_c = self.model.transform(v1, v2)
        return X_c, Y_c

if __name__ == "__main__":
    data = data_generate()
    clf_ = cca

    # gene data
    name = ['Srbct', 'Leukemia', 'Lymphoma', 'Prostate', 'Brain', 'Colon']

    i = 0
    data.generate_genes_data(num=i)

    print()
    print("finish reading data: ", name[i])
    print()

    # train gcca model
    clf = clf_(ds=data, m_rank=2)
    clf.solve()

    # calculate all kind of metric
    v1_test, v2_test = clf.transform(data.test_data)
    print("total correlation in training data is: ", np.mean(clf.cal_correlation(clf.list_projection)))
    print("total correlation in testing data is: ", np.mean(clf.cal_correlation([v1_test, v2_test])))
    print("training data ACC is: ", clf.cal_acc(clf.list_projection))
    print("testing data ACC is: ", clf.cal_acc([v1_test, v2_test]))
    print("each view's spare of U is ", clf.cal_spare())
    print("total sqare is: ", clf.cal_spare()[0])

    print()
    print()