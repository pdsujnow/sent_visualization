#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import itertools
from collections import Counter

import inspect

from keras.preprocessing import sequence
from keras.models import Sequential, Graph
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.layers import containers

class BaseWrapper(object):
    def __init__(self, **model_params):
        optimizer='adam' if 'optimizer' not in model_params else model_params['optimizer']
        loss_fn='categorical_crossentropy' if 'loss_fn' not in model_params else model_params['loss_fn']
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        model_params.update(self.get_default_params())
        self.model_params = model_params
        for name, value in model_params.items():
            setattr(self, name, value)


    def get_params(self, deep=True):
        res = {}
        arg_name = ['optimizer', 'loss_fn']
        res = {name: getattr(self, name) for name in arg_name}

        for (name, value) in self.model_params.items():
            res.update({name: value})
        return res

    def set_params(self, **params):
        for name, value in params.items():
            setattr(self, name, value)
        self.construct()
        self.compile(self.optimizer, self.loss_fn)
        return self


    def get_default_params(self):
        raise NotImplementedError

    def construct(self):
        raise NotImplementedError

class SequentialWrapper(BaseWrapper):

    def fit(self, X, y):
        assert(type(self.model)==Sequential)

        fit_params = {}
        fit_params_name = inspect.getargspec(Sequential.fit)
        for name,value in self.model_params.items():
            if name in fit_params_name:
                fit_params.update({name: value})

        return self.model.fit(X, y, **fit_params)

class CNN(SequentialWrapper):
    def get_default_params(self):
        return dict(
        maxlen = 100,
        vocabulary_size = None,
        drop_out_prob = 0.2,
        embedding_dims = 100,
        nb_filter = 250,
        hidden_dims = 100,
        filter_length = [2,3,4]
        )


    def construct(self):
        
        #np.random.seed(random_state)

        self.model = Sequential()
        self.model.add(Embedding(self.vocabulary_size, self.embedding_dims, input_length=self.maxlen))
        #self.model.add(Dropout(drop_out_prob))
        self.model.add(self.convLayer(self.maxlen, self.embedding_dims, self.nb_filter, self.filter_length))
        self.model.add(Dense(hidden_dims))
        #self.model.add(Dropout(drop_out_prob))
        self.model.add(Activation('relu'))
        self.model.add(Dense(y.shape[1]))
        self.model.add(Activation('sigmoid'))

    def convLayer(self, inp_dim, embedding_dims, nb_filter, filter_length):
        c = containers.Graph()
        c.add_input(name='input', input_shape=(inp_dim, embedding_dims))
        inps=[]
        for i in filter_length:
            c.add_node(containers.Sequential([
                                Convolution1D(nb_filter=nb_filter, filter_length=i, border_mode='valid', activation='relu',subsample_length=1, input_shape=(inp_dim, embedding_dims)),
                                MaxPooling1D(pool_length=2),
                                Flatten()]),
                                name='Conv{}'.format(i), input='input')
            inps.append('Conv{}'.format(i))

        if len(inps)==1:
            c.add_output('output', input=inps[0])
        else:
            c.add_output('output', inputs=inps)

        return c

    #def init(self):
    #    if 'arch_file' not in self.__dict__:
    #        return
    #    with open(self.arch_file) as f:
    #        self.model = model_from_json(f.read())
    #    self.model.load_weights(self.weight_file)

    #def un_init(self):
    #    del self.model

    #def dump_to_file(self, dump_file):
    #    self.arch_file = dump_file+'_arch.json'
    #    self.weight_file = dump_file+'_weights.h5'
    #    with open(self.arch_file, 'w') as f:
    #        f.write(self.model.to_json())
    #    self.model.save_weights(self.weight_file)
    #    super(CNN, self).dump_to_file(dump_file)
