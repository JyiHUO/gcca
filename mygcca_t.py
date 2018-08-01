import numpy as np
import sklearn.datasets as ds
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from utils import *
import pandas as pd
import scipy.io as sco


class GeneralizedCCA:

    def __init__(self, ds, m_rank=0):
        '''
        Constructor for GeneralizedCCA.

        Args:
            list_view (list<ndarray>): Training data for each view
            m_rank (int): How many principal components to keep. A value of 0
                indicates that it should be full-rank. (Default 0)
        '''

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

    def solve_u(self):
        number_of_views = len(self.list_view)

        # cal G
        self.solve_g()
        # print (self.G.shape)

        for i in range(number_of_views):
            U = np.linalg.pinv(self.list_view[i].transpose()) * np.mat(self.G)

            projected_data = np.mat(self.list_view[i].transpose()) * np.mat(U)

            self.list_U.append(np.array(U))
            self.list_projection.append(np.array(projected_data))



    def solve_u_linearized_bregman(self, verbose = True):
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


    def cal_correlation(self, rank=True):
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


class data_generate:
    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        self.origin_train_data = None
        self.train_data = None  # [(D, N), (D, N) ... ]
        self.test_data = None

    def generate_boston(self, normalize=True):
        boston = ds.load_boston()
        X = boston.data
        y = boston.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6,random_state=10)

        self.origin_train_data = [X_train.T, y_train.reshape((-1, 1)).T]

        if normalize:
            X_train, y_train = self._center_norm(X_train, y_train)

        self.train_data = [X_train.T, y_train.reshape((-1, 1)).T]
        self.test_data = [X_test.T, y_test.reshape((-1, 1)).T]

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

        self.origin_train_data = [x_train1.T, x_train2.T]


        self.train_data = [x_train1.T, x_train2.T]
        self.test_data = [x_test1.T, x_test2.T]
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

        self.origin_train_data = [y_train1.T, y_test1.T]

        self.train_data = [x_train1.T, y_train.T]
        self.test_data = [x_test1.T, y_test.T]

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
            return left.reshape((N, -1)).T, right.reshape((N, -1)).T

        self.train_data = _cut_half(x_train)
        self.test_data = _cut_half(x_test)

    def en_es_fr(self, row = 800, normalize = False):
        if row == 800:
            v1 = pd.read_csv("csv_data/en800data.csv", encoding = "ISO-8859-1").values
            v2 = pd.read_csv("csv_data/es800data.csv", encoding = "ISO-8859-1").values
        else:
            v1 = pd.read_csv("csv_data/en1000data.csv", encoding = "ISO-8859-1").values
            v2 = pd.read_csv("csv_data/fr1000data.csv", encoding = "ISO-8859-1").values

        if normalize:
            self._center_norm(v1, v2)
        v1_train, v1_test, v2_train, v2_test = train_test_split(v1, v2, test_size = 0.8, random_state = 42)

        self.train_data = [v1_train.T, v2_train.T]
        self.test_data = [v1_test.T, v2_test.T]

    def generate_genes_data(self,num=0, normalize=False):
        Srbct = sco.loadmat("../gcca_data/genes_data/Srbct.mat")
        Leukemia = sco.loadmat("../gcca_data/genes_data/Leukemia.mat")
        Lymphoma = sco.loadmat("../gcca_data/genes_data/Lymphoma.mat")
        Prostate = sco.loadmat("../gcca_data/genes_data/Prostate.mat")

        data = [Srbct, Leukemia, Lymphoma, Prostate]

        v1 = data[num]["fea"]
        v2 = pd.get_dummies(data[num]["gnd"].reshape((-1))).values

        print(v1.shape)
        print(v2.shape)

        self.origin_train_data = [v1, data[num]["gnd"].reshape((-1))]
        if normalize:
            v1, v2 = self._center_norm(v1, v2)

        v1_train, v1_test, v2_train, v2_test = train_test_split(v1, v2, test_size=0.6, random_state=42)

        self.train_data = [v1_train.T, v2_train.T]
        self.test_data = [v1_test.T, v2_test.T]


    def _center_norm(self, X, y):
        N = X.shape[0]

        self.x_mean = X.mean(0)
        self.x_std = X.std(0)
        self.y_mean = y.mean(0)
        self.y_std = y.std(0)

        X = X.reshape((N, -1))
        y = y.reshape((N, -1))
        X = (X - X.mean(0)) / X.std(0)
        y = (y - y.mean(0)) / y.std(0)

        return X, y

if __name__ == "__main__":
    # generate boston data
    dg = data_generate()
    # dg.generate_boston()
    # dg.generate_mnist(normalize=False)
    # dg.generate_mnist_half()
    # dg.en_es_fr(800)
    dg.generate_genes_data(num=0)

    gcca = GeneralizedCCA(ds=dg, m_rank=2)
    gcca.solve_u()
    # gcca.solve_u_linearized_bregman(verbose=True)
    print(gcca.cal_correlation())
    print(gcca.cal_acc(gcca.list_projection))
    print(gcca.cal_acc(gcca.transform(dg.test_data[0].T, dg.test_data[1].T)))
    print(gcca.cal_spare())
    print(np.mean(gcca.cal_spare()))
    print("训练集：", dg.train_data[0].shape)
    print("测试集：", dg.test_data[1].shape)