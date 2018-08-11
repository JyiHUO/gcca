import os

from dgcca_tool.dgcca import DGCCAArchitecture, LearningParams, DGCCA
from metric import *
import theano
import theano.tensor as T


class dgcca_(metric):
    def __init__(self, ds, m_rank):
        super().__init__()
        self.list_view = ds.train_data  # [np.float32(d) for d in ds.train_data]   # [(N, D), (N, D) ... ]
        self.ds = ds
        self.list_U = []  # save U for each view [(D, r), (D, r) ... ]
        self.list_projection = []  # save project data through U for each view [(N, r), (N, r) ... ]
        self.m_rank = m_rank

        self.model = None

    def solve(self):
        viewMlpStruct = [[v.shape[1], 10, 10, 10, self.m_rank] for v in self.list_view]  # Each view has single-hidden-layer MLP with slightly wider hidden layers

        # Actual data used in paper plot...

        # self.K = np.ones((400, 3), dtype=np.float32)

        arch = DGCCAArchitecture(viewMlpStruct, self.m_rank, 2, activation=T.nnet.relu)

        # Little bit of L2 regularization -- learning params from matlab synthetic experiments
        lparams = LearningParams(rcov=[0.01] * 3, viewWts=[1.0] * 3, l1=[0.0] * 3, l2=[5.e-4] * 3,
                                 optStr='{"type":"adam","params":{"adam_b1":0.1,"adam_b2":0.001}}',
                                 batchSize=40,
                                 epochs=200)
        vnames = ['View1', 'View2', 'View3']

        model = DGCCA(arch, lparams, vnames)
        model.build()

        history = []


        history.extend(model.learn(self.list_view, tuneViews=None, trainMissingData=None,
                                               tuneMissingData=None, embeddingPath=None,
                                               modelPath=None, logPath=None, calcGMinibatch=False))

        # self.train_G = model.apply(self.list_view, isTrain=True)
        self.model = model
        self.list_U = []


    def cal_G_error(self, list_view, test = True):
        K = np.ones((list_view[0].shape[0], 3), dtype=np.float32)
        return self.model.reconstructionErr(list_view, missingData=K)