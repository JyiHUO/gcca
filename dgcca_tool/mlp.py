"""
Lifted from Theano multilayer perceptron tutorial.  Adapted to take the partial derivative of
some external loss w.r.t. output layer and back-propagate this loss.  Output layer has no
nonlinearity.

Adrian Benton
9/13/2016
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import os
import sys
import timeit

import numpy as np
from numpy.random import RandomState

import theano
from   theano.gradient import jacobian
import theano.tensor as T

from functools import reduce

import unittest

theano.config.compute_test_value = 'off' # Use 'warn' to activate this feature, 'off' otherwise

# Test values. . .
np.random.seed(12345)
sample_n_examples = 400
sample_n_hidden = [ 50, 10, 5 ]
sample_input = np.random.randn(sample_n_examples, sample_n_hidden[0])
sample_externalGrad  = np.random.randn(sample_n_examples, sample_n_hidden[-1])

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh, includeBias=False, vname=''):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)
        
        :type n_in: int
        :param n_in: dimensionality of input
        
        :type n_out: int
        :param n_out: number of hidden units
        
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        
        :type includeBias: bool
        :param includeBias: Whether this layer should have a bias term
        
        :type vname: str
        :param vname: name to attach to this layer's weights
        """
        
        self.input = input
        
        # Weight initialization for different nonlinearities
        if W is None:
          if activation == T.nnet.sigmoid:
            W_values = 4. * np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out))
                , dtype=theano.config.floatX
            )
          elif activation == T.nnet.relu:
            W_values = np.asarray(
                rng.normal(
                    0.0,
                    2.0/n_in,
                    size=(n_in, n_out))
                , dtype=theano.config.floatX
            )
          else:
            # Xavier initialization for tanh
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out))
                , dtype=theano.config.floatX
            )
          W = theano.shared(value=W_values.astype(theano.config.floatX),
                            name='%s_W' % (vname), borrow=True)
        
        if b is None:
          b_values = np.zeros((n_out,), dtype=theano.config.floatX)
          b = theano.shared(value=b_values.astype(theano.config.floatX),
                            name='%s_b' % (vname), borrow=True)
        
        self.W = W
        self.b = b
        
        if includeBias:
          lin_output = T.dot(input, self.W) + self.b
          self.params = [self.W, self.b]
        else:
          lin_output = T.dot(input, self.W)
          self.params = [self.W]
        
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

class MLPWithLinearOutput(object):
    """Multi-Layer Perceptron Class
    
    This has no softmax at the output layer, just a stack of layers with
    nonlinearities -- final layer has no non-linearity, and then a linear projection
    to shared space.
    """
    
    def __init__(self, seed, layerWidths, activation, gccaDim, optimizer, L1_reg=0.0, L2_reg=0.0, vname=''):
        """Initialize the parameters for the multilayer perceptron
        
        :type seed: int
        :param seed: to init random number generator
        
        :type layerWidths: [ int ]
        :param layerWidths: width of each layer
        
        :type activation: Tensor -> Tensor
        :param activation: activation function for hidden layers
        
        :type gccaDim: int
        :param gccaDim: dimensionality of shared layer
        
        :type optimizer: Optimizer
        :param optimizer: generates weight updates
        
        :type L1_reg: float
        :param L1_reg: weight to place on L1 penalty on weights
        
        :type L2_reg: float
        :param L2_reg: weight to place on L2 penalty
        
        :type vname: str
        :param vname: name to attach to layer weights
        """
        
        rng = RandomState(seed)
        self.optimizer = optimizer
        
        self.L1_reg = theano.shared(np.array(L1_reg).astype(theano.config.floatX),
                                    'L1_%s' % (vname))
        self.L2_reg = theano.shared(np.array(L2_reg).astype(theano.config.floatX),
                                    'L2_%s' % (vname))
        
        # Learned externally by GCCA routine
        self.U = theano.shared(np.random.randn(layerWidths[-1],
                                               gccaDim).astype(theano.config.floatX),
                               'U_%s' % (vname), allow_downcast=True)
        
        self.__hiddenLayers = []
        
        self.input  = T.matrix('X_%s' % (vname))
        self.missing = T.vector('K_missing_%s' % (vname)) # to compute mean of output layer when we have missing data
        self.__externalGrad = T.matrix('ExternalGrad_%s' % (vname)) # Partial derivative of loss w.r.t. output layer -- computed somewhere else
        
        self.input.tag.test_value = sample_input
        self.__externalGrad.tag.test_value = sample_externalGrad
        
        Ws = []
        
        # Connect hidden layers
        for layerIndex, (nIn, nOut) in enumerate(zip(layerWidths, layerWidths[1:])):
          prevLayer = self.input if layerIndex == 0 else self.__hiddenLayers[-1].output
          
          act = activation if layerIndex < ( len(layerWidths) - 2 ) else None
          
          hiddenLayer = HiddenLayer(
            rng=rng,
            input=prevLayer,
            n_in=nIn,
            n_out=nOut,
            activation=act,
            includeBias=True,
            vname='%s_layer-%d'  % (vname, layerIndex)
          )
          
          self.__hiddenLayers.append(hiddenLayer)
          Ws.append(hiddenLayer.W)
          
          if layerIndex == 0:
            self.L1     = abs(hiddenLayer.W).sum()
            self.L2_sqr = (hiddenLayer.W ** 2).sum()
          else:
            self.L1 += abs(hiddenLayer.W).sum()
            self.L2_sqr += (hiddenLayer.W ** 2).sum()
        
        # L1/L2 regularization terms
        self.__reg_cost = (
          self.L1_reg * self.L1
          + self.L2_reg * self.L2_sqr
        )
        
        # Mean-centers the output layer.  Calculated on training data.
        self.Bcenter = theano.shared(np.zeros((1, layerWidths[-1])).
                                     astype(theano.config.floatX),
                                     name='%s_BmeanCenter' % (vname),
                                     broadcastable=(True, False), borrow=True,
                                     allow_downcast=True)
        
        self.__output_uncentered = self.__hiddenLayers[-1].output
        self.output_centered     = self.__output_uncentered - self.Bcenter
        self.shared_output       = self.output_centered.dot(self.U)
        
        mask  = T.tile(self.missing.reshape((self.output_centered.shape[0], 1)),
                                              (1,self.output_centered.shape[1]))
        denom = 1./mask.sum(axis=0, keepdims=True)
        
        # Recenter based on current training data
        self.__Bcenter_current = (mask * self.__output_uncentered).sum(axis=0,
                                                                       keepdims=True) * denom
        self.output_traindata_centered = self.__output_uncentered - self.__Bcenter_current
        
        # so we can update all parameters at once
        self.__params = reduce(lambda x,y: x+y,
                             [layer.params for layer in self.__hiddenLayers])
        
        # Hack to get theano autodiff to compute and backprop gradients for me.
        # Idea from Nanyun Peng.
        self.__external_cost = T.sum( self.output_centered * self.__externalGrad )
        
        self.__cost = self.__reg_cost + self.__external_cost
        
        # Gradient for just the external loss.
        self.__gparams = [T.grad(self.__external_cost, p) for p in self.__params]
        
        self.__reg_gparams = [T.grad(self.__reg_cost, p) for p in Ws]
        
        # Full gradient update
        self.__full_gparams = [T.grad(self.__cost, p) for p in self.__params]
        
        self.buildFns()
    
    def getWeights(self):
      wts = [p.get_value() for p in self.__params]
      return wts
    
    def setWeights(self, weights):
      '''
      Parameters
      ----------
      :type weights: [ np.array ]
      :param weights: should be the same number of elements and shapes as self.__params
      '''
      
      for param, wts in zip(self.__params, weights):
        param.set_value(np.float32(wts))
    
    def buildFns(self):
      # What to call when applying to test
      self.get_shared_output = theano.function(
        inputs=[self.input],
        outputs=self.shared_output
          , allow_input_downcast=True)
      
      # What to call when training
      self.get_centered_output = theano.function(
        inputs=[self.input],
        outputs=self.output_centered
          , allow_input_downcast=True)
      
      self.get_uncentered_output = theano.function(
        inputs=[self.input],
        outputs=self.__output_uncentered
          , allow_input_downcast=True)
      
      # Different cost and gradient functions for inspection/debugging.
      self.calc_gradient      = theano.function( inputs=[ self.input, self.__externalGrad ],
                                                 outputs=self.__gparams , allow_input_downcast=True)
      self.calc_regOnly_gradient = theano.function( inputs=[], outputs=self.__reg_gparams, allow_input_downcast=True)
      self.calc_reg_gradient  = theano.function( inputs=[ self.input, self.__externalGrad ],
                                                 outputs=self.__full_gparams , allow_input_downcast=True)
      self.calc_external_cost = theano.function( inputs=[ self.input, self.__externalGrad ],
                                                 outputs=self.__external_cost , allow_input_downcast=True)
      self.calc_reg_cost      = theano.function( inputs=[ ], outputs=self.__reg_cost , allow_input_downcast=True)
      self.calc_total_cost       = theano.function( inputs=[ self.input,
                                                             self.__externalGrad ],
                                                    outputs=self.__cost , allow_input_downcast=True)
      
      # Gradient w.r.t. output layer
      self.calc_gradient_wrtOutput = theano.function( 
        inputs=[ self.input, self.__externalGrad ],
        outputs=T.grad(self.__cost, self.output_centered)
          , allow_input_downcast=True)
      self.calc_gradient_wrtOutputUC = theano.function(
        inputs=[ self.input, self.__externalGrad ],
        outputs=T.grad(self.__cost, self.__output_uncentered)
          , allow_input_downcast=True)
      
      # For debugging, get hidden layer values
      self.get_layer_values = theano.function(inputs=[ self.input ],
                                              outputs=[h.output for h in self.__hiddenLayers]
                                              , allow_input_downcast=True)
      
      if self.optimizer is not None:
        self.setOptimizer(self.optimizer)
      
      self.Bupdate = (self.Bcenter, self.__Bcenter_current)
      
      # Set B to the mean of these data, and return output
      self.center_output = theano.function( inputs=[ self.input, self.missing ],
                                            outputs=self.output_traindata_centered,
                                            updates=[self.Bupdate] , allow_input_downcast=True)
  
    def setOptimizer(self, optimizer):
      self.optimizer = optimizer
      self.__updates = self.optimizer.getUpdates(self.__params, self.__full_gparams)
      updates = self.__updates
      params  = self.__params
      grads   = self.__full_gparams
      
      self.take_step = theano.function([ self.input, self.__externalGrad ],
                                       outputs=[],
                                       updates=self.__updates, allow_input_downcast=True)
