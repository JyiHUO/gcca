from metric import *
from data_class import *
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import theano.tensor as T
from keras.callbacks import ModelCheckpoint
from utils import load_data, svm_classify
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.io as sco
from keras.layers import Dense, Merge
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.regularizers import l2


class deepcca(metric):
    def __init__(self, ds, m_rank, batch_size = 50, epoch_num = 10, learning_rate = 1e-3):
        super().__init__()
        self.list_view = ds.train_data  # [(D, N), (D, N) ... ]
        self.ds = ds
        self.list_U = []  # save U for each view [(D, r), (D, r) ... ]
        self.list_projection = []  # save project data through U for each view [(N, r), (N, r) ... ]
        self.m_rank = m_rank

        self.model = None


        # parameter you can tune
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate

    def solve(self):
        # the path to save the final learned features
        save_to = './new_features.gz'

        v1, v2 = self.list_view

        # the size of the new space learned by the model (number of the new features)
        outdim_size = self.m_rank

        # size of the input for view 1 and view 2
        input_shape1 = v1.shape[1]  # 784
        input_shape2 = v2.shape[1]  # 784

        # number of layers with nodes in each one
        layer_sizes1 = [1024, 1024, 1024, outdim_size]
        layer_sizes2 = [1024, 1024, 1024, outdim_size]

        # the parameters for training the network
        learning_rate = self.learning_rate
        epoch_num = self.epoch_num
        batch_size = self.batch_size

        # the regularization parameter of the network
        # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
        reg_par = 1e-5

        # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
        # if one option does not work for a network or dataset, try the other one
        use_all_singular_values = False

        # if a linear CCA should get applied on the learned features extracted from the networks
        # it does not affect the performance on noisy MNIST significantly
        apply_linear_cca = True

        # end of parameters section

        model = self.create_model(layer_sizes1, layer_sizes2, input_shape1, input_shape2,
                             learning_rate, reg_par, outdim_size, use_all_singular_values)
        # model.summary()
        self.model = self.train_model(model, v1, v2, epoch_num, batch_size)

        self.list_projection = self.t_model(model, v1, v2, outdim_size, apply_linear_cca)

    def train_model(self, model, data1, data2, epoch_num, batch_size):
        """
        trains the model
        # Arguments
            data1 and data2: the train, validation, and test data for view 1 and view 2 respectively. data should be packed
            like ((X for train, Y for train), (X for validation, Y for validation), (X for test, Y for test))
            epoch_num: number of epochs to train the model
            batch_size: the size of batches
        # Returns
            the trained model
        """

        # Unpacking the data
        train_set_x1, valid_set_x1, train_set_x2, valid_set_x2 = train_test_split(data1, data2, test_size=0.1,
                                                                                  random_state=42)

        # best weights are saved in "temp_weights.hdf5" during training
        # it is done to return the best model based on the validation loss

        # used dummy Y because labels are not used in the loss function
        model.fit([train_set_x1, train_set_x2], np.zeros(len(train_set_x1)),
                  batch_size=batch_size, epochs=epoch_num, shuffle=True,
                  validation_data=([valid_set_x1, valid_set_x2], np.zeros(len(valid_set_x1))), verbose=0)

        results = model.evaluate([valid_set_x1, valid_set_x2], np.zeros(len(valid_set_x1)), batch_size=batch_size,
                                 verbose=0)
        # print('loss on validation data: ', results)
        return model

    def cal_spare(self):
        return [0,0]

    def t_model(self, model, data1, data2, outdim_size, apply_linear_cca):
        """produce the new features by using the trained model
        # Arguments
            model: the trained model
            data1 and data2: the train, validation, and test data for view 1 and view 2 respectively.
                Data should be packed like
                ((X for train, Y for train), (X for validation, Y for validation), (X for test, Y for test))
            outdim_size: dimension of new features
            apply_linear_cca: if to apply linear CCA on the new features
        # Returns
            new features packed like
                ((new X for train - view 1, new X for train - view 2, Y for train),
                (new X for validation - view 1, new X for validation - view 2, Y for validation),
                (new X for test - view 1, new X for test - view 2, Y for test))
        """

        # producing the new features

        new_data = []

        pred_out = model.predict([data1, data2])  # (50000, 20)
        r = int(pred_out.shape[1] / 2)  # 10
        new_data = [pred_out[:, :r], pred_out[:, r:]]

        # based on the DCCA paper, a linear CCA should be applied on the output of the networks because
        # the loss function actually estimates the correlation when a linear CCA is applied to the output of the networks
        # however it does not improve the performance significantly
        if apply_linear_cca:
            w = [None, None]
            m = [None, None]
            # print("Linear CCA started!")
            w[0], w[1], m[0], m[1] = self.linear_cca(new_data[0], new_data[1], outdim_size)  # weight (10, 10)
            # print("Linear CCA ended!")

            # Something done in the original MATLAB implementation of DCCA, do not know exactly why;)
            # it did not affect the performance significantly on the noisy MNIST dataset
            # s = np.sign(w[0][0,:])
            # s = s.reshape([1, -1]).repeat(w[0].shape[0], axis=0)
            # w[0] = w[0] * s
            # w[1] = w[1] * s
            ###
            data_num = len(new_data[0])
            for v in range(2):
                new_data[v] -= m[v].reshape([1, -1]).repeat(data_num, axis=0)  # center data before prediction
                new_data[v] = np.dot(new_data[v], w[v])  # do some prediction
            self.list_U = w

        return new_data

    def transform(self, list_view):
        '''
        :param v1: (N, D)
        :param v2:
        :return:
        '''
        v1, v2 = list_view
        new_data = self.t_model(self.model, v1, v2, self.m_rank, apply_linear_cca=True)
        return new_data

    def linear_cca(self, H1, H2, outdim_size):
        """
        An implementation of linear CCA
        # Arguments:
            H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
            outdim_size: specifies the number of new features
        # Returns
            A and B: the linear transformation matrices
            mean1 and mean2: the means of data for both views
        """
        r1 = 1e-4
        r2 = 1e-4

        m = H1.shape[0]
        o = H1.shape[1]

        mean1 = np.mean(H1, axis=0)
        mean2 = np.mean(H2, axis=0)
        H1bar = H1 - np.tile(mean1, (m, 1))
        H2bar = H2 - np.tile(mean2, (m, 1))

        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(o)
        SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(o)

        [D1, V1] = np.linalg.eigh(SigmaHat11)
        [D2, V2] = np.linalg.eigh(SigmaHat22)
        SigmaHat11RootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

        Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        [U, D, V] = np.linalg.svd(Tval)
        V = V.T
        A = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
        B = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
        D = D[0:outdim_size]

        return A, B, mean1, mean2

    def create_model(self, layer_sizes1, layer_sizes2, input_size1, input_size2,
                     learning_rate, reg_par, outdim_size, use_all_singular_values):
        """
        builds the whole model
        the structure of each sub-network is defined in build_mlp_net,
        and it can easily get substituted with a more efficient and powerful network like CNN
        """
        view1_model = self.build_mlp_net(layer_sizes1, input_size1, reg_par)
        view2_model = self.build_mlp_net(layer_sizes2, input_size2, reg_par)

        model = Sequential()
        model.add(Merge([view1_model, view2_model], mode='concat'))

        model_optimizer = RMSprop(lr=learning_rate)
        model.compile(loss=self.cca_loss(outdim_size, use_all_singular_values), optimizer=model_optimizer)

        return model

    def build_mlp_net(self, layer_sizes, input_size, reg_par):
        model = Sequential()
        for l_id, ls in enumerate(layer_sizes):
            if l_id == 0:
                input_dim = input_size
            else:
                input_dim = []
            if l_id == len(layer_sizes) - 1:
                activation = 'linear'
            else:
                activation = 'sigmoid'

            model.add(Dense(ls, input_dim=input_dim,
                            activation=activation,
                            kernel_regularizer=l2(reg_par)))
        return model

    def cca_loss(self, outdim_size, use_all_singular_values):
        """
        The main loss function (inner_cca_objective) is wrapped in this function due to
        the constraints imposed by Keras on objective functions
        """

        def inner_cca_objective(y_true, y_pred):
            """
            It is the loss function of CCA as introduced in the original paper. There can be other formulations.
            It is implemented by Theano tensor operations, and does not work on Tensorflow backend
            y_true is just ignored, because y_true = np.zeros(len(train_set_x1))
            y_pred = [train_set_x1, train_set_x2]
            """

            r1 = 1e-4
            r2 = 1e-4
            eps = 1e-12
            o1 = o2 = y_pred.shape[1] // 2

            # unpack (separate) the output of networks for view 1 and view 2
            print(y_pred)
            H1 = y_pred[:, 0:o1].T  # (10, N)
            H2 = y_pred[:, o1:o1 + o2].T

            m = H1.shape[1]

            H1bar = H1 - (1.0 / m) * T.dot(H1, T.ones([m, m]))
            H2bar = H2 - (1.0 / m) * T.dot(H2, T.ones([m, m]))

            SigmaHat12 = (1.0 / (m - 1)) * T.dot(H1bar, H2bar.T)
            SigmaHat11 = (1.0 / (m - 1)) * T.dot(H1bar, H1bar.T) + r1 * T.eye(o1)
            SigmaHat22 = (1.0 / (m - 1)) * T.dot(H2bar, H2bar.T) + r2 * T.eye(o2)

            # Calculating the root inverse of covariance matrices by using eigen decomposition
            [D1, V1] = T.nlinalg.eigh(SigmaHat11)
            [D2, V2] = T.nlinalg.eigh(SigmaHat22)

            # Added to increase stability
            posInd1 = T.gt(D1, eps).nonzero()[0]
            D1 = D1[posInd1]
            V1 = V1[:, posInd1]
            posInd2 = T.gt(D2, eps).nonzero()[0]
            D2 = D2[posInd2]
            V2 = V2[:, posInd2]

            SigmaHat11RootInv = T.dot(T.dot(V1, T.nlinalg.diag(D1 ** -0.5)), V1.T)
            SigmaHat22RootInv = T.dot(T.dot(V2, T.nlinalg.diag(D2 ** -0.5)), V2.T)

            Tval = T.dot(T.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

            if use_all_singular_values:
                # all singular values are used to calculate the correlation
                corr = T.sqrt(T.nlinalg.trace(T.dot(Tval.T, Tval)))
            else:
                # just the top outdim_size singular values are used
                [U, V] = T.nlinalg.eigh(T.dot(Tval.T, Tval))
                U = U[T.gt(U, eps).nonzero()[0]]
                U = U.sort()
                corr = T.sum(T.sqrt(U[0:outdim_size]))

            return -corr

        return inner_cca_objective

if __name__ == "__main__":
    data = data_generate()
    clf_ = deepcca

    # gene data
    name = ['Srbct', 'Leukemia', 'Lymphoma', 'Prostate', 'Brain', 'Colon']

    i = 0
    data.generate_genes_data(num=i)

    print()
    print("finish reading data: ", name[i])
    print()

    # train deepcca model
    clf = clf_(ds=data, m_rank=3, batch_size = 50, epoch_num = 10, learning_rate = 1e-3)
    clf.solve()

    # calculate all kind of metric
    v1_test, v2_test = clf.transform(data.test_data)
    print("total correlation in training data is: ", np.sum(clf.cal_correlation(clf.list_projection)))
    print("total correlation in testing data is: ", np.sum(clf.cal_correlation([v1_test, v2_test])))
    print("training data ACC is: ", clf.cal_acc(clf.list_projection))
    print("testing data ACC is: ", clf.cal_acc([v1_test, v2_test]))
    print("each view's spare of U is ", clf.cal_spare())
    #print("total sqare is: ", clf.cal_spare()[0])

    print()
    print()

