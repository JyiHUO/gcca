'''
Attempt at reimplementing deep GCCA.  Supports missing views.

Adrian Benton
9/13/2016
'''

from functools import reduce

import theano
import theano.tensor as T

import numpy as np

import math, os, random, sys, time

from .mlp import MLPWithLinearOutput
from . import opt
from .wgcca import WeightedGCCA

class DGCCAArchitecture:
  '''
  Specifies architecture of DGCCA network.
  '''
  
  def __init__(self, viewMlps, gccaDim, truncParam=1000, activation=T.nnet.relu):
    '''
    Parameters
    ----------
    viewMlps: [ [ int ] ]
        Each element is an MLP for a particular view, each integer is the layer width
        (first layer being input, last layer is penultimate to shared multiview layer.)
    gccaDim: int
        Dimensionality of multiview representation
    truncParam: int
        Quality of low-rank approximation to GCCA data matrices.
    activation: [ Theano function: Tensor -> Tensor ]
        Activation function for layers.
    '''
    
    self.viewMlps = viewMlps
    self.gccaDim  = gccaDim
    self.truncParam = truncParam
    self.activation = activation

class LearningParams:
  '''
  Parameters affecting how we learn this model.
  '''
  
  def __init__(self, rcov, viewWts, l1, l2, batchSize, epochs, optStr, valFreq=1):
    '''
    Parameters
    ----------
    rcov: [ float ]
        Small amount of regularization to apply to covariance matrix for each
        projected view.
    viewWts: [ float ]
        How much to weight each view in GCCA objective.  Also weights learaning when
        backpropagating gradient.
    l1: [ float ]
        L1 regularization applied uniformly to network weights per view
    l2: [ float ]
        L2 regularization on all weights
    batchSize: int
        Number of example per minibatch.  As-of now, minibatches are only used
        to pass the data through feedforward networks, and backprop.  GCCA
        is still performed batch.
    epochs: int
        How long to train the model.
    valFreq: int
        Number of epochs between checking validation error.
    optStr: str
        JSON-serialized optimizer.  See opt.jsonToOpt for how this is deserialized.
    '''
    
    self.rcov    = rcov
    self.viewWts = viewWts
    self.l1      = l1
    self.l2      = l2
    self.batchSize = batchSize
    self.epochs    = epochs
    self.valFreq   = valFreq
    self.optStr    = optStr

class DGCCAModel:
  '''
  Theano computation graph.  The GCCA step at the end falls back on
  a scipy implementation.
  '''
  
  def __init__(self, architecture, rcov=1.e-12, viewWts=1.0, viewNames=None, seed=-1):
    self.V      = len(architecture.viewMlps)
    self.arch   = architecture
    self.vnames = viewNames if (viewNames is not None) else ['View%d' % (i) for i, mlp in enumerate(self.arch.viewMlps)]
    
    if viewWts is None:
      self.viewWts = [1.0 for i in range(self.V)]
    elif type(viewWts) == float:
      self.viewWts = [viewWts for i in range(self.V)]
    else:
      self.viewWts = viewWts
    
    # random seed to initialize network weights
    self._seed  = seed
    
    if self._seed > -1:
      np.random.seed(self._seed)
    
    # Needed to get U and G to calculate gradient -- external to Theano graph
    self.gccaModule = WeightedGCCA( self.V,
                                    [ lwidths[-1] for lwidths in self.arch.viewMlps ],
                                    self.arch.gccaDim,
                                    rcov, self.arch.truncParam, viewWts=self.viewWts,
                                    verbose=False )
    
    self.Us = [] # Keep track of weights mapping from network output layer to multiview layer.
    self.Bs = [] # Bias terms in final layer -- used to mean-center projections.
    
    self._build()
  
  def _build(self):
    ### Build feedforward networks  ###
    self.nets           = []
    self.inputs         = []
    self.shared_outputs = []
    K_missing           = []
    for lwidths, vname in zip(self.arch.viewMlps, self.vnames):
      net = MLPWithLinearOutput(np.random.randint(10**2, 10**5), lwidths,
                                self.arch.activation, self.arch.gccaDim,
                                None, L1_reg=0.0, L2_reg=0.0, vname=vname)
      self.nets.append(net)
      self.Us.append(net.U)
      self.Bs.append(net.Bcenter)
      
      # Handles to networks' in/out -- needed to compile our own functions
      self.inputs.append(net.input)
      self.shared_outputs.append(net.shared_output)
      K_missing.append(net.missing)
    
    weightedK = [w * K for K, w in zip(K_missing, self.viewWts) ]
    Kdenom = 1./T.sum(weightedK, axis=0)
    Kdenom = Kdenom.reshape((weightedK[0].shape[0], 1))
    
    Gprimes = []
    Gprimes_zeroed = []
    
    ### Construct theano function to map new data to shared space ###
    for Gprime, K_i in zip(self.shared_outputs, weightedK):
      Gprimes.append( Gprime )
      Gprimes_zeroed.append( K_i.reshape((K_i.shape[0], 1)) * Gprime)
      #Gprimes_zeroed.append( T.tile(K_i.reshape((K_i.shape[0], 1)),
      #                              (1,Gprime.shape[1])) * Gprime)
    
    self.__Gmean   = (Kdenom * (T.sum( Gprimes_zeroed, axis=0)) )
    self.__Gprimes = T.stack(Gprimes) # 3-tensor ( V x N x k )
    
    # Set B, centering term, based on train, and return output layers
    self.center_outputs   = theano.function(inputs=self.inputs + K_missing,
                                    outputs=[net.output_traindata_centered
                                             for net in self.nets],
                                    updates=[net.Bupdate for net in self.nets], allow_input_downcast=True)
    self.get_outputs_centered = theano.function(inputs=self.inputs,
                                                outputs=[net.output_centered
                                                         for net in self.nets]
                                                , allow_input_downcast=True)
    
    # These are used to calculate reconstruction error -- how close are view-specific
    # projections to G
    self.getViewProjs = theano.function(self.inputs, outputs=self.__Gprimes, allow_input_downcast=True)
    
    # Get average projection or projections of each view
    self.getGmean = theano.function(self.inputs + K_missing, outputs=self.__Gmean, allow_input_downcast=True)
  
  def setWeights(self, wts):
    '''
    For loading pretrained model
    
    Parameters
    ----------
    wts : [ [ np.array ] ]
          Weights for each view network.  Last two matrices are U, B for multiview step
    '''
    
    for net, netWts, U, B in zip(self.nets, wts, self.Us, self.Bs):
      net.setWeights( [w.astype(theano.config.floatX) for w in netWts[:-2]] )
      U.set_value( netWts[-2].astype(theano.config.floatX) )
      B.set_value( netWts[-1].astype(theano.config.floatX) )
  
  def getWeights(self):
    '''
    For saving model
    
    Returns
    -------
    wts : [ [ np.array ] ]
          List of weights for each view network.  Can be passed as argument to setWeights
    '''
    
    wts = []
    for net, U, B in zip(self.nets, self.Us, self.Bs):
      netWts = net.getWeights()
      netWts += [U.get_value(), B.get_value()]
      wts.append(netWts)
    
    return wts
  
  def apply(self, views, missingData=None, isTrain=False):
    '''
    Get multiview embedding of our data.  Project each view through the
    network, then take mean embedding of all views.
    '''
    
    if isTrain: # compute mean-center term on these data
      outputs = self.center_outputs(*(views + [missingData[:,i] for i
                                               in range(missingData.shape[1])]))
      self.gccaModule.learn(outputs, missingData, incremental=False)
      return self.gccaModule.G
    else:
      _K = [missingData[:,i] for i in range(missingData.shape[1])]
      return self.getGmean(*(views + _K))
  
  def _prepNetForTrain(self, lparams):
    '''
    Set regularization and learning parameters before training.
    '''
    
    self.lparams = lparams
    
    # Making sure to set these before training, otherwise, no regularization is applied...
    l1 = lparams.l1
    l2 = lparams.l2
    
    if (type(l1) == float) or (type(l1)==np.ndarray and len(l1.shape)==0):
      l1 = [l1 for i in range(len(self.nets))]
    if (type(l2) == float) or (type(l2)==np.ndarray and len(l2.shape)==0):
      l2 = [l2 for i in range(len(self.nets))]
    
    for net, l1val, l2val in zip(self.nets, l1, l2):
      net.L1_reg.set_value( np.array(l1val).astype(theano.config.floatX) )
      net.L2_reg.set_value( np.array(l2val).astype(theano.config.floatX) )
    
    for w, net in zip(self.viewWts, self.nets):
      optimizer = opt.jsonToOpt(lparams.optStr)
      
      # Adjust learning rate based on view weighting
      try:
        optimizer.learningRate = optimizer.learningRate * w
      except Exception as ex:
        print ('Cannot adjust learning rate of optimizer "%s" for view "%s"' %
               (optimizer.__class__.__name__, net.vname))
      
      net.setOptimizer(optimizer)
  
  def train(self, lparams, trainViews, trainMissingData, tuneViews=None, tuneMissingData=None, logger=None, calcGMinibatch=False):
    '''
    Trains network and GCCA linear maps.  If tuning data is provided, will keep
    network parameters that minimize reconstruction error.  Yields the current train
    and validation error every lparams.valFreq epochs.
    
    Parameters
    ----------
    lparams: LearningParams
        Learning parameters.
    trainViews: [ matrix ]
        Feature vectors for train.  Each view is an element in this list.
    trainMissingData: matrix
        Encodes which views we are missing data for, in train
    tuneViews: [ matrix ]
        Feature vectors for tuning set
    tuneMissingData: matrix
        Missing views in tune
    logger: Logger
        If not None, logs parameters after each update
    calcGMinibatch: bool
        If true, calculates G, U for minibatches, only.
    '''
    
    # If None, assume we have no missing data in views
    trainMissingData = trainMissingData if trainMissingData is not None else np.ones( (trainViews[0].shape[0], len(trainViews) ) )
    if tuneViews is not None:
      tuneMissingData = tuneMissingData if tuneMissingData is not None else np.ones( (tuneViews[0].shape[0], len(tuneViews) ) )
    
    self._prepNetForTrain(lparams)
    
    # SGD parameters
    bsize = lparams.batchSize # minibatch size, only used in backprop
    
    try:
      if len(lparams.rcov) == len(trainViews):
        self.gccaModule.eps = [np.array(e).astype(theano.config.floatX)
                               for e in lparams.rcov]
      else:
        self.gccaModule.eps = [np.array(lparams.rcov).astype(theano.config.floatX)
                               for i in range(len(trainViews))]
    except Exception as ex:
      self.gccaModule.eps = [np.array(lparams.rcov).astype(theano.config.floatX)
                             for i in range(len(trainViews))]
    
    # Because we backprop w/ SGD, need to split up data into minibatches.
    # We reshuffle order in which minibatch steps are taken every epoch.
    minibatch_indices = []
    for i in range(0, math.ceil(trainViews[0].shape[0]/bsize) ):
      sidx = i*bsize
      eidx = sidx + bsize
      minibatch_indices.append( (sidx, eidx) )
    
    random.shuffle(minibatch_indices)
    
    epochs  = lparams.epochs
    valFreq = lparams.valFreq
    
    # Main training loop
    for epoch in range(epochs):
      startTime = time.time()
      
      # Feedforward to output layer & mean-center
      outputs = self.center_outputs(*(trainViews +
                                      [trainMissingData[:,i] for i
                                       in range(trainMissingData.shape[1])]))
      
      # Solve for G, U
      if not calcGMinibatch:
        self.gccaModule.learn(outputs, trainMissingData, incremental=False)
        trainG, trainLbda, Us = self.gccaModule.G, self.gccaModule.lbda, self.gccaModule.U
        for Ushared, Uval in zip(self.Us, Us):
          Ushared.set_value( Uval.astype(theano.config.floatX) )
      
      for sidx, eidx in minibatch_indices:
        if calcGMinibatch:
          self.gccaModule.learn([o[sidx:eidx,:] for o in outputs],
                                trainMissingData[sidx:eidx,:], incremental=False)
          trainG, trainLbda, Us = self.gccaModule.G, self.gccaModule.lbda, self.gccaModule.U
          for Ushared, Uval in zip(self.Us, Us):
            Ushared.set_value( Uval.astype(theano.config.floatX) )
        
        # Update each network's weights in minibatches
        for Vidx, (V, output, net, U) in enumerate(zip(trainViews, outputs, self.nets, self.Us)):
          Uval = U.get_value()
          
          G = trainG if calcGMinibatch else trainG[sidx:eidx,:]
          
          #grad = 2.0*( trainG[sidx:eidx,:] - output[sidx:eidx,:].dot(Uval) ).dot(Uval.T)
          grad = ( G - output[sidx:eidx,:].dot(Uval) ).dot(Uval.T)
          
          grad *= trainMissingData[sidx:eidx,Vidx].reshape((grad.shape[0], 1)) # Don't backprop for missing views
          
          net.take_step( V[sidx:eidx,:], grad )
      
      random.shuffle( minibatch_indices )
      
      # Calculate reconstruction error
      if not ( (epoch+1) % valFreq ):
        trainErr = self.reconstructionErr(trainViews, trainMissingData, trainG)
        if tuneViews is not None:
          tuneErr = self.reconstructionErr(tuneViews, tuneMissingData)
        else:
          tuneErr = float('nan')
        
        endTime = time.time()
        epochTime = endTime - startTime
        
        yield epoch, trainErr, tuneErr, epochTime
      
      ## Can be used for logging parameters, etc.
      if logger is not None:
        logger.log(self, epoch, lr, trainG, trainErr, tuneErr)
      
      #print('Completed epoch %d' % (epoch+1))
    
    outputs = self.center_outputs(*(trainViews +
                                    [trainMissingData[:,i] for i
                                     in range(trainMissingData.shape[1])]))
    self.gccaModule.learn(outputs, trainMissingData, incremental=False)
    trainG, trainLbda, Us = self.gccaModule.G, self.gccaModule.lbda, self.gccaModule.U
    for Ushared, Uval in zip(self.Us, Us):
      Ushared.set_value( Uval.astype(theano.config.floatX) )
    
    # Check reconstruction error one last time. . .
    trainErr = self.reconstructionErr(trainViews, trainMissingData, trainG)
    if tuneViews is not None:
      tuneErr = self.reconstructionErr(tuneViews, tuneMissingData)
    else:
      tuneErr = float('nan')
    
    endTime = time.time()
    epochTime = endTime - startTime
    
    yield epoch, trainErr, tuneErr, epochTime
  
  def reconstructionErr(self, views, missingData=None, G=None):
    '''
    Evaluate this model on how well the mappings U from output layer can construct G.
    If G is provided, then this is training reconstruction error.
    '''
    
    Gprimes = self.getViewProjs(*views)
    
    if G is None:
      # Just take the mean of projected views, this is for tuning data
      decompK = [missingData[:,i] for i in range(missingData.shape[1])]
      G = self.getGmean(*(views + decompK))
    
    #import pdb; pdb.set_trace()
    
    # Penalize views based on weighting
    wts_tensor = [[[w]] for w in self.viewWts]
    temp = wts_tensor * missingData.T.reshape((missingData.shape[1],missingData.shape[0], 1)) *(G - Gprimes)
    r_err = np.linalg.norm(temp)**2

    # r_err = np.sum(np.mean(wts_tensor * missingData.T.reshape((missingData.shape[1],missingData.shape[0], 1)) *np.abs(G - Gprimes), axis=0))
    
    return r_err

class ParamLogger:
  def __init__(self, logPath='test.npz'):
    self.logPath = logPath
    self.history = []
  
  def log(self, model, epoch, lr, trainG, trainErr, tuneErr):
    outDict = []
    outDict.extend([('epoch', epoch), ('learning_rate', lr),
                    ('trainErr', trainErr), ('tuneErr', tuneErr)])
    
    self.history.append(outDict)
    
    np.savez_compressed(self.logPath, self.history)

class DGCCA:
  def __init__(self, architecture, learningParams, viewNames=None):
    '''
    Parameters
    ----------
    architecture: DGCCAArchitecture
        Defines network architecture.  Dimensionality of each layer for each view. 
    learningParams: LearningParams
        Affects how we train this network.
    viewNames: [ str ]
        Name of each view.  Used to name network layers.
    '''
    
    self.arch    = architecture
    self.lparams = learningParams
    self.vnames  = viewNames
  
  def build(self, initWeights=None, randSeed=12345):
    '''
    Parameters
    ----------
    initWeights: [ [ np array ] ]
        Weights to initialize network with.  Each element in outer list
        corresponds to a view, each value is a weight matrix from layer
        $i$ to $i+1$ or bias term (with last two elements being the U
        and mean-centering bias).
    randSeed: float
        Seed to generate random weights.  If None, uses clock time.
    '''
    
    randSeed = randSeed if randSeed is not None else int(time.time())
    
    self._model = DGCCAModel(self.arch, self.lparams.rcov,
                             self.lparams.viewWts, self.vnames,
                             randSeed)
    
    if initWeights is not None:
      print('Initializing weights!')
      self._model.setWeights(initWeights)
  
  def save(self, fpath):
    '''
    Writes model to compressed numpy file.  Cannot serialize functions, so
    layer activations are stored as string.
    '''
    
    activationStr = 'sigmoid'
    # Defaults to sigmoid if we don't know this nonlinearity
    if self.arch.activation == T.nnet.relu:
      activationStr = 'relu'
    elif self.arch.activation == T.nnet.sigmoid:
      activationStr = 'sigmoid'
    elif self.arch.activation == T.nnet.tanh:
      activationStr = 'tanh'
    else:
      activationStr = 'sigmoid'
    
    ser = {'activation':activationStr,
           'arch_viewMlps':self.arch.viewMlps,
           'arch_gccaDim':self.arch.gccaDim,
           'lparam_rcov':self.lparams.rcov,
           'lparam_viewWts':self.lparams.viewWts,
           'lparam_l1':self.lparams.l1,
           'lparam_l2':self.lparams.l2,
           'lparam_opt':self.lparams.optStr,
           'lparam_bsize':self.lparams.batchSize,
           'lparam_epochs':self.lparams.epochs,
           'lparam_valFreq':self.lparams.valFreq,
           'weights':self._model.getWeights(),
           'viewNames':self.vnames}
    
    np.savez_compressed(fpath, **ser)
  
  @staticmethod
  def load(fpath):
    ''' Loads model from file.  '''
    
    model_desc = np.load(fpath)
    
    if 'activation' not in model_desc:
      print ('No activation specified -- assuming sigmoid')
      activation = T.nnet.sigmoid
    elif model_desc['activation'] == 'sigmoid':
      activation = T.nnet.sigmoid
    elif model_desc['activation'] == 'relu':
      activation = T.nnet.relu
    elif model_desc['activation'] == 'tanh':
      activation = T.nnet.tanh
    else:
      activation = lambda x: x # identity....
      print ('Initializing network w/o nonlinearities, do not recognize: %s' %
             (activationStr))
    
    if 'lparam_viewWts' in model_desc:
      viewWts = model_desc['lparam_viewWts']
    else:
      viewWts = [1.0 for i in range(len(model_desc['arch_viewMlps']))]
    
    architecture = DGCCAArchitecture(viewMlps=model_desc['arch_viewMlps'],
                                     gccaDim=model_desc['arch_gccaDim'],
                                     activation=activation)
    lparams      = LearningParams(rcov=model_desc['lparam_rcov'],
                                  viewWts=viewWts,
                                  l1=model_desc['lparam_l1'], l2=model_desc['lparam_l2'], 
                                  batchSize=model_desc['lparam_bsize'].item(),
                                  epochs=model_desc['lparam_epochs'].item(),
                                  optStr=model_desc['lparam_opt'].item(),
                                  valFreq=model_desc['lparam_valFreq'].item())
    weights      = model_desc['weights']
    vnames       = model_desc['viewNames']
    
    model = DGCCA(architecture, lparams, vnames)
    model.build(weights)
    
    return model
  
  def learn(self, trainViews, tuneViews=None, trainMissingData=None, tuneMissingData=None, embeddingPath=None, modelPath=None, logPath=None, calcGMinibatch=False):
    '''
    Train the network
    
    Parameters
    ----------
    trainViews: [ np array ( numExamples, numFeatures_j ) ]
        Data to learn model on
    trainMissingData: np array ( numExamples, numViews )
        Ignores views for which data are missing
    tuneViews: [ np array ( numExamples, numFeatures_j ) ]
        Data used for tuning.  If tuning data is None, then we'll just keep the weights in the last epoch
    tuneMissingData: np array ( numExamples, numViews )
        Ignores example views with missing data in tune
    embeddingPath: str
        Path to store concatenation of train+tune embeddings.  If None, they are not stored.
    modelPath: str
        Path to model checkpoint file.  Should be .npz
    logPath: str
        Path to write out train/tune reconstruction error, etc.
    calcGMinibatch: bool
        If true, computes U,G w.r.t. a minibatch instead of whole training set
    '''
    
    # If None, assume we have no missing data in views
    trainMissingData = trainMissingData if trainMissingData is not None else np.ones( (trainViews[0].shape[0], len(trainViews) ) )
    if tuneViews is not None:
      tuneMissingData = tuneMissingData if tuneMissingData is not None else np.ones( (tuneViews[0].shape[0], len(tuneViews) ) )
    
    lowestTuneError = float('inf')
    
    history = []
    
    # Train model, logging reconstruction error every valFreq epochs
    for epoch, trainErr, tuneErr, epochTime in self._model.train(self.lparams, trainViews, trainMissingData, tuneViews, tuneMissingData, logger=None, calcGMinibatch=calcGMinibatch):
      print('Epoch %d (%ds) -- train_err: %f val_err: %f' %
            (epoch, epochTime, trainErr, tuneErr))
      sys.stdout.flush()
      
      history.append((epoch, trainErr, tuneErr))
      
      # Best tuning loss so far, or no tuning set provided, checkpoint model and error
      if np.isnan(tuneErr) or tuneErr < lowestTuneError:
        if modelPath is not None:
          self.save(modelPath)
          print('Checkpointed model')
        
        if logPath is not None:
          history.append((epoch, trainErr, tuneErr))
          np.savez_compressed(logPath, history=history)
          print('Saved history')
        
        if embeddingPath is not None:
          trainEmbedding = self.apply(trainViews, trainMissingData)
          if tuneViews is not None:
            tuneEmbedding  = self.apply(tuneViews,  tuneMissingData)
            G = np.vstack([trainEmbedding, tuneEmbedding])
          else:
            G = trainEmbedding
          
          np.savez_compressed(embeddingPath, G=G)
          
          print('Saved embeddings')
        
        lowestTuneError = tuneErr
    
    return history
  
  def apply(self, views, missingData=None, isTrain=False):
    return self._model.apply(views, missingData, isTrain=isTrain)
  
  def reconstructionErr(self, views, missingData=None):
    return self._model.reconstructionErr(views, missingData)
  
if __name__ == '__main__':
  pass
