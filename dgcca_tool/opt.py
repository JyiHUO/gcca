'''
Optimizers for DGCCA network.

Adrian Benton
10/17/2016
'''

import json
import numpy as np

import theano
import theano.tensor as T

class Optimizer:
  def getUpdates(self, params, grads):
    '''
    :type params: [ theano_shared_variable ]
    :param params: model weights
    
    :type grads: [ theano_tensor ]
    :param grads: symbolic gradient w.r.t each set of weights
    
    :returns: [ (theano_shared_variable, theano_tensor) ] -- updated weights/history
              after a single gradient step
    '''
    raise NotImplementedError
  
  def reset(self):
    '''
    To reset history, etc.
    '''
    raise NotImplementedError
  
  def toJson(self):
    raise NotImplementedError

class SGDOptimizer(Optimizer):
  ''' Vanilla SGD with decay. '''
  
  def __init__(self, learningRate=0.01, decay=1.0):
    '''
    :type learningRate: float
    :param learningRate: how big a step to take each epoch
    '''
    self.learningRate = theano.shared(np.array(learningRate).astype(theano.config.floatX))
    self.decay        = theano.shared( np.array(decay).astype(theano.config.floatX) )
  
  def getUpdates(self, params, grads):
    updates = []
    
    for p, g in zip(params, grads):
      step = self.learningRate * g
      updates.append( (p, p + step) )
    
    updates.append( ( self.learningRate, self.learningRate*self.decay ) )
    
    return updates
  
  def reset(self):
    pass
  
  def toJson(self):
    return json.dumps({'type':'sgd',
                       'params':{'learningRate':self.learningRate.get_value(),
                                 'decay':self.decay.get_value()}})

class MomentumSGDOptimizer(Optimizer):
  ''' SGD with momentum term. '''
  
  def __init__(self, learningRate=0.01, momentum=0.0, decay=1.0):
    '''
    :type learningRate: float
    :param learningRate: how big a step to take each epoch
    
    :type momentum: float
    :param momentum: how badly to go in the same direction
    '''
    self.learningRate = theano.shared( np.array(learningRate).astype(theano.config.floatX) )
    self.momentum     = theano.shared( np.array(momentum).astype(theano.config.floatX) )
    self.decay        = theano.shared( np.array(decay).astype(theano.config.floatX) )
    
    self.__prevSteps = None # will initialize these to zero once we know the shape of weights
  
  def getUpdates(self, params, grads):
    if self.__prevSteps is None:
      self.__prevSteps = []
      for p in params:
        self.__prevSteps.append( theano.shared(p.get_value()*np.array(0.).
                                               astype(theano.config.floatX),
                                             allow_downcast=True) )
    
    updates = []
    
    for prevStep, p, g in zip(self.__prevSteps, params, grads):
      momentumPrev = self.momentum*prevStep
      sgdStep = self.learningRate * g
      step = momentumPrev + sgdStep
      
      updates.append( (prevStep, step) )
      updates.append( (p, p + step) )
    
    updates.append( ( self.learningRate, self.learningRate*self.decay ) )
    
    return updates
  
  def toJson(self):
    return json.dumps({'type':'sgd_momentum',
                       'params':{'learningRate':self.learningRate.get_value(),
                                 'momentum':self.momentum.get_value(),
                                 'decay':self.decay.get_value()}})

class AdamOptimizer(Optimizer):
  '''
  Adam, adaptive learning rate optimization: https://arxiv.org/pdf/1412.6980v8.pdf
  Implementation based on code in https://gist.github.com/Newmu/acb738767acb4788bac3
  '''
  
  def __init__(self, learningRate=0.01, adam_b1=0.1, adam_b2=0.001):
    '''
    :type  learningRate: float
    :param learningRate: how big a step to take each epoch
    
    :type  adam_b1: float
    :param adam_b1: 1 - decay rate for first moment estimate
    
    :type  adam_b2: float
    :param adam_b2: 1 - decay rate for second moment estimate
    '''
    self.learningRate = theano.shared(np.array(learningRate).astype(theano.config.floatX))
    
    self.adam_b1 = theano.shared( np.array(adam_b1).astype(theano.config.floatX) )
    self.adam_b2 = theano.shared( np.array(adam_b2).astype(theano.config.floatX) )
    
    self.__adam_i   = theano.shared( np.array(0.0).astype(theano.config.floatX) )
    self.__adam_i_t = self.__adam_i + np.array(1.0).astype(theano.config.floatX)
    
    self.__moments_m = None
    self.__moments_v = None
  
  def getUpdates(self, params, grads):
    updates = []
    
    # Moment bias correction
    fix1 = 1. - (1. - self.adam_b1)**self.__adam_i_t
    fix2 = 1. - (1. - self.adam_b2)**self.__adam_i_t
    lr_t = self.learningRate * (T.sqrt(fix2)/fix1)
    
    updates = []
    
    if (self.__moments_m is None) or (self.__moments_v is None):
      self.__moments_m = []
      self.__moments_v = []
      
      for p, g in zip(params, grads):
        self.__moments_m.append( theano.shared(p.get_value() *
                                               np.array(0.).astype(theano.config.floatX),
                                 allow_downcast=True) )
        self.__moments_v.append( theano.shared(p.get_value() *
                                               np.array(0.).astype(theano.config.floatX),
                                 allow_downcast=True) )
      
      for p, g, adam_m, adam_v in zip(params, grads, self.__moments_m, self.__moments_v):
        adam_m_t = (self.adam_b1 * g) + ((1. - self.adam_b1) * adam_m)
        adam_v_t = (self.adam_b2 * T.sqr(g)) + ((1. - self.adam_b2) * adam_v)
        step = lr_t * adam_m_t / (T.sqrt(adam_v_t) + np.float32(1.e-8))
        
        updates.append((adam_m, adam_m_t))
        updates.append((adam_v, adam_v_t))
        updates.append((p,  p + step))
      
    updates.append((self.__adam_i, self.__adam_i_t))
    
    return updates
  
  def reset(self):
    self.__adam_i.set_value(np.float32(0.0))
    
    for moments in [self.__moments_m, self.__moments_v]:
      if moments is not None:
        for moment in moments:
          moment.set_value( moment.get_value()*0.0 )
  
  def toJson(self):
    return json.dumps({'type':'sgd_momentum',
                       'params':{'learningRate':self.learningRate.get_value(),
                                 'adam_b1':self.adam_b1.get_value(),
                                 'adam_b2':self.adam_b2.get_value()}})

def jsonToOpt(jsonStr):
  ''' Build optimizer from JSON string. '''
  
  optObj = json.loads(jsonStr)
  
  optType   = optObj['type']
  optParams = optObj['params']
  
  if optType == 'sgd':
    return SGDOptimizer(**optParams)
  elif optType == 'sgd_momentum':
    return MomentumSGDOptimizer(**optParams)
  elif optType == 'adam':
    return AdamOptimizer(**optParams)
  else:
    raise Exception('Optimizer type "%s" not in {sgd, sgd_momentum, adam}' % (optType))
