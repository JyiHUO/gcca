from sklearn.cross_decomposition import CCA
import numpy as np
import pandas as pd
from metric import *
from data_class import *

class cca(metric):
    def __init__(self, ds, m_rank):
        super().__init__()
        self.list_view = ds.train_data  # [(D, N), (D, N) ... ]
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
        self.list_U = [clf.x_weights_, clf.y_weights_]

    def transform(self, v1, v2):
        '''
        :param v1: (N, D)
        :param v2:
        :return:
        '''
        X_c, Y_c = self.model.transform(v1, v2)
        return X_c, Y_c

if __name__ == "__main__":
    # generate boston data
    dg = data_generate()
    # dg.generate_boston()
    # dg.generate_mnist(normalize=False)
    # dg.generate_mnist_half()
    # dg.en_es_fr(800)
    dg.generate_genes_data(num=0)
    # dg.generate_twitter_dataset()

    print()
    print("finish reading data")
    print()

    cca = cca(ds=dg, m_rank=50)
    # gcca.solve_u()
    cca.solve()
    # print(gcca.cal_correlation())
    # print(gcca.cal_acc(gcca.list_projection))
    # print(gcca.cal_acc(gcca.transform(dg.test_data[0].T, dg.test_data[1].T)))
    print(cca.cal_spare())
    print(np.mean(cca.cal_spare()))
    print("训练集：", dg.train_data[0].shape)
    print("测试集：", dg.test_data[1].shape)