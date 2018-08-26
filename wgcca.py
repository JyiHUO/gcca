import pickle, gzip, os, sys, time

import numpy as np
import scipy
import scipy.sparse
import scipy.linalg

import argparse

import pickle
from data_class import *
from metric import *


class WeightedGCCA(metric):
    '''
    Weighted generalized canonical correlation analysis (WGCCA).
    Implemented with batch SVD.
    '''

    def __init__(self, ds, m_rank, verbose=True):
        super().__init__()
        self.V = 3  # Number of views
        self.F = [d.shape[1] for d in ds.train_data]  # Number of features per view
        self.k = m_rank  # Dimensionality of embedding we want to learn
        eps = [1.e-8 for i in range(self.V)]
        viewWts = [np.random.rand() for i in range(3)]

        # some common setting
        self.list_view = ds.train_data
        self.ds = ds
        self.m_rank = m_rank
        self.list_U = []  # save U for each view [(D, r), (D, r) ... ]
        self.list_projection = []  # save project data through U for each view [(N, r), (N, r) ... ]

        # Regularization for each view
        try:
            if len(eps) == self.V:
                self.eps = [np.float32(e) for e in eps]
            else:
                self.eps = [np.float32(eps) for i in range(self.V)]  # Assume eps is same for each view
        except:
            self.eps = [np.float32(eps) for i in range(self.V)]  # Assume eps is same for each view

        self.W = [np.float32(v) for v in viewWts] if viewWts else [np.float32(1.) for v in range(self.V)]  # How much we should weight each view -- defaults to equal weighting

        self.U = None  # Projection from each view to shared space
        self.G = None  # Embeddings for training examples
        self.G_scaled = None  # Scaled by SVs of covariance matrix sum

        self.verbose = verbose

    def _compute(self, views, K=None, incremental=False):
        '''
        Compute G by first taking low-rank decompositions of each view, stacking
        them, then computing the SVD of this matrix.
        '''

        # K ignores those views we have no data for.  If it is not provided,
        # then we use all views for all examples.  All we need to know is
        # K^{-1/2}, which just weights each example based on number of non-zero
        # views.  Will fail if there are any empty examples.
        if K is None:
            K = np.float32(np.ones((views[0].shape[0], len(views))))
        else:
            K = np.float32(K)

        # We do not want to count missing views we are downweighting heavily/zeroing out, so scale K by W
        K = K.dot(np.diag(self.W))
        Ksum = np.sum(K, axis=1)

        # If we have some missing rows after weighting, then make these small & positive.
        Ksum[Ksum == 0.] = 1.e-8

        K_invSqrt = scipy.sparse.dia_matrix((1. / np.sqrt(Ksum), np.asarray([0])), shape=(K.shape[0], K.shape[0]))

        # Left singular vectors for each view along with scaling matrices

        As = []
        Ts = []
        Ts_unnorm = []

        N = views[0].shape[0]

        _Stilde = np.float32(np.zeros(self.k))
        _Gprime = np.float32(np.zeros((N, self.k)))
        _Stilde_scaled = np.float32(np.zeros(self.k))
        _Gprime_scaled = np.float32(np.zeros((N, self.k)))

        # Take SVD of each view, to calculate A_i and T_i
        for i, (eps, view) in enumerate(zip(self.eps, views)):
            A, S, B = scipy.linalg.svd(view, full_matrices=False, check_finite=False)

            # Find T by just manipulating singular values.  Matrices are all diagonal,
            # so this should be fine.

            S_thin = S[:self.k]

            S2_inv = 1. / (np.multiply(S_thin, S_thin) + eps)

            T = np.diag(
                np.sqrt(
                    np.multiply(np.multiply(S_thin, S2_inv), S_thin)
                )
            )

            # Keep singular values
            T_unnorm = np.diag(S_thin + eps)

            if incremental:
                ajtj = K_invSqrt.dot(np.sqrt(self.W[i]) * A.dot(T))
                ajtj_scaled = K_invSqrt.dot(np.sqrt(self.W[i]) * A.dot(T_unnorm))

                _Gprime, _Stilde = WeightedGCCA._batch_incremental_pca(ajtj,
                                                                       _Gprime,
                                                                       _Stilde)
                _Gprime_scaled, _Stilde_scaled = WeightedGCCA._batch_incremental_pca(ajtj_scaled,
                                                                                     _Gprime_scaled,
                                                                                     _Stilde_scaled)
            else:
                # Keep the left singular vectors of view j
                As.append(A[:, :self.k])
                Ts.append(T)
                Ts_unnorm.append(T_unnorm)

            if self.verbose:
                print('Decomposed data matrix for view %d' % (i))

        if incremental:
            self.G = _Gprime
            self.G_scaled = _Gprime_scaled

            self.lbda = _Stilde
            self.lbda_scaled = _Stilde_scaled
        else:
            # In practice M_tilde may be really big, so we would
            # like to perform this SVD incrementally, over examples.
            M_tilde = K_invSqrt.dot(np.bmat([np.sqrt(w) * A.dot(T) for w, A, T in zip(self.W, As, Ts)]))

            Q, R = scipy.linalg.qr(M_tilde, mode='economic')

            # Ignore right singular vectors
            U, lbda, V_toss = scipy.linalg.svd(R, full_matrices=False, check_finite=False)

            self.G = Q.dot(U[:, :self.k])
            self.lbda = lbda

            # Unnormalized version of G -> captures covariance between views
            M_tilde = K_invSqrt.dot(np.bmat([np.sqrt(w) * A.dot(T) for w, A, T in zip(self.W, As, Ts_unnorm)]))
            Q, R = scipy.linalg.qr(M_tilde, mode='economic')

            # Ignore right singular vectors
            U, lbda, V_toss = scipy.linalg.svd(R, full_matrices=False, check_finite=False)

            self.lbda_scaled = lbda
            self.G_scaled = self.G.dot(np.diag(self.lbda_scaled[:self.k]))

            if self.verbose:
                print('Decomposed M_tilde / solved for G')

        self.U = []  # Mapping from views to latent space
        self.U_unnorm = []  # Mapping, without normalizing variance
        self._partUs = []

        # Now compute canonical weights
        for idx, (eps, f, view) in enumerate(zip(self.eps, self.F, views)):
            R = scipy.linalg.qr(view, mode='r')[0]
            Cjj_inv = np.linalg.inv((R.transpose().dot(R) + eps * np.eye(f)))
            pinv = Cjj_inv.dot(view.transpose())

            self._partUs.append(pinv)

            self.U.append(pinv.dot(self.G))
            self.U_unnorm.append(pinv.dot(self.G_scaled))

            if self.verbose:
                print('Solved for U in view %d' % (idx))

    @staticmethod
    def _batch_incremental_pca(x, G, S):
        r = G.shape[1]
        b = x.shape[0]

        xh = G.T.dot(x)
        H = x - G.dot(xh)
        J, W = scipy.linalg.qr(H, overwrite_a=True, mode='full', check_finite=False)

        Q = np.bmat([[np.diag(S), xh], [np.zeros((b, r), dtype=np.float32), W]])

        G_new, St_new, Vtoss = scipy.linalg.svd(Q, full_matrices=False, check_finite=False)
        St_new = St_new[:r]
        G_new = np.asarray(np.bmat([G, J]).dot(G_new[:, :r]))

        return G_new, St_new

    def learn(self, views, K=None, incremental=False):
        '''
        Learn WGCCA embeddings on training set of views.  Set incremental to true if you have
        many views.
        '''

        self._compute(views, K, incremental)
        return self

    def apply(self, views, K=None, scaleBySv=False):
        '''
        Extracts WGCCA embedding for new set of examples.  Maps each present view with
        $U_i$ and takes mean of embeddings.

        If scaleBySv is true, then does not normalize variance of each canonical
        direction.  This corresponds to GCCA-sv in "Learning multiview embeddings
        of twitter users."  Applying WGCCA to a single view with scaleBySv set to true
        is equivalent to PCA.
        '''

        Us = self.U_unnorm if scaleBySv else self.U
        projViews = []

        N = views[0].shape[0]

        if K is None:
            K = np.ones((N, self.V))  # Assume we have data for all views/examples

        for U, v in zip(Us, views):
            projViews.append(v.dot(U))
        projViewsStacked = np.stack(projViews)

        # Get mean embedding from all views, weighting each view appropriately

        weighting = np.multiply(K, self.W)  # How much to weight each example/view

        # If no views are present, embedding for that example will be NaN
        denom = weighting.sum(axis=1).reshape((N, 1))

        # Weight each view
        weightedViews = weighting.T.reshape((self.V, N, 1)) * projViewsStacked

        Gsum = weightedViews.sum(axis=0)
        Gprime = Gsum / denom

        Gprime = np.multiply(Gsum, 1. / denom)

        return Gprime

    def solve(self):
        self.learn(self.list_view)
        self.list_U = self.U
        self.list_projection = self.transform(self.list_view)

if __name__ == "__main__":
    data = data_generate()
    clf_ = WeightedGCCA
#
#    # three views data for tfidf language data
#
    # data.generate_three_view_tfidf_dataset()
    #
    # clf = clf_(ds=data, m_rank=20)
    # clf.solve()
    #
    # # calculate all kind of metric
    # print("reconstruction error of G in training is: ", clf.cal_G_error(data.train_data, test=False))
    # print("reconstruction error of G in testing is: ", clf.cal_G_error(data.test_data, test=True))
    # print("each view's spare of U is ", clf.cal_spare())
    # #print("total sqare is: ", np.mean(clf.cal_spare()))
    #
    # print()
    # print()

    # for synthetic data
    data.generate_synthetic_dataset()

    clf = clf_(ds=data, m_rank=1)
    clf.solve()

    # calculate all kind of metric
    print("reconstruction error of G in training is: ", clf.cal_G_error(data.train_data, test=False))
    print("reconstruction error of G in testing is: ", clf.cal_G_error(data.test_data, test=True))
    print("each view's spare of U is ", clf.cal_spare())
    print("total sqare is: ", np.mean(clf.cal_spare()))

    print()
    print()
    clf.save_U("wgcca_synthetic")
