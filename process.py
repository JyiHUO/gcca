# -*-coding=utf-8-*-
import sklearn.datasets as ds
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.linalg import svd
from sklearn import datasets as ds
from sklearn.metrics import r2_score

def generate_data():
    v1 = np.ones((1000, 1))
    v1[200:700, :] = -1
    v1[700:, :] = 0

    v2 = np.ones((1500, 1))
    v2[:800, :] = 0
    v2[800:1300:, :] = 1
    v2[1300:, :] = -1

    miu1 = np.random.normal(loc=0, scale=0.3, size=(1000, 100))
    miu2 = np.random.normal(loc=0, scale=0.3, size=(1500, 100))

    u1 = np.random.normal(loc=0, scale=1, size=(100, 1))
    u2 = np.random.normal(loc=0, scale=1, size=(100, 1))

    train = v1.dot(u1.T) + miu1
    target = v2.dot(u2.T) + miu2


    return [train, target]  # (D, N)

def generate_boston():
    boston = ds.load_boston()
    train = boston.data
    target = boston.target

    return [train.T, target.reshape((-1, 1)).T]


def cal_A_B(list_view, num_row):
    '''
    Calculate common space of G and some necessary variable
    :param list_view: [view1, view2 ...] view shape:(D, N)
    :param num_row: top r eig vector
    :return: matrix G, list A , list B
    '''
    N = list_view[0].shape[1]
    G = np.zeros((N, N))

    A = []
    B = []
    S = []

    for i, view in enumerate(list_view):
        p, s, q = np.linalg.svd(view, full_matrices=False)
        G += q.T.dot(q)

        A.append(p)
        B.append(q)
        S.append(s)

    _, _, G_eig = np.linalg.svd(G)

    G = G_eig[:num_row, :]

    # calculate A and B
    for i in range(len(A)):

        n = S[i].shape[0]
        sigama = np.zeros((n, n))
        sigama[np.arange(n), np.arange(n)] = S[i]
        A[i] = A[i].T
        B[i] = np.linalg.inv(sigama).dot(B[i].dot(G.T))


    return A, B

def linearized_bregman(A, B, verbose = True):
    '''
    Solve equation which is Ax = B
    :param A: matrix
    :param B: matrix
    :return: matrix X
    '''

    # initialize parameter
    error_x = 1
    epsilon = 1e-5
    delta = 0.9
    tau = 1
    mu_x = 5
    Numit_x = 0
    Vx_tilde = A.T.dot(B)
    Vx_old = Vx_tilde

    # solve X
    X = None
    while error_x > epsilon:
        X = delta * np.sign(Vx_tilde) * np.maximum(tau * np.abs(Vx_tilde) - mu_x, 0)
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

def cal_U(list_view, top_r):
    A, B = cal_A_B(list_view, top_r)

    list_U = []

    for i in range(len(B)):
        list_U.append(linearized_bregman(A[i], B[i]))

        print()
        print("next view: ")
        print()

    return list_U

def cov_matrix(list_view, list_U, select = False):
    left = list_U[0].T.dot(list_view[0])  # (top_r, N)
    right = list_U[1].T.dot(list_view[1])

    up = left.dot(right.T)
    down1 = left.dot(left.T)
    down2 = right.dot(right.T)

    temp = up / np.square(down1 * down2)
    if select:
        n = temp.shape[0]
        return temp[np.arange(n), np.arange(n)]
    else:
        return temp


def cal_sparse(U):
    return np.sum(U == 0) / (U.shape[0] * U.shape[1])

def cal_r2_score(list_view, list_U):
    left = list_U[0].T.dot(list_view[0])  # (top_r, N)
    right = list_U[1].T.dot(list_view[1])

    return r2_score(left, right)

def main():
    list_view = generate_data()
    top_r = 20

    A, B = cal_A_B(list_view, top_r)

    list_U = []


    for i in range(len(B)):
        list_U.append(linearized_bregman(A[i], B[i]))

    print(cov_matrix(list_view, list_U))

    # print(A[0].shape)
    # print(B[0].shape)
    # print(list_U[0].shape)  # (D, top_r)

if __name__ == "__main__":
    main()