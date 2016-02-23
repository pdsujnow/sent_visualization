#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential, Graph
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import containers

def build_cnn( maxlen = 10,
        vocabulary_size = None,
        nb_class = 1,
        drop_out_prob = 0.25,
        embedding_dims = 128,
        nb_filter = 250,
        hidden_dims = 250,
        filter_length = [3,4]):

    optimizer='adam'
    loss_fn='categorical_crossentropy'
    #np.random.seed(random_state)

    model = Sequential()
    print vocabulary_size, embedding_dims, maxlen
    model.add(Embedding(vocabulary_size, embedding_dims, input_length=maxlen))
    #model.add(Dropout(drop_out_prob))
    model.add(convLayer(maxlen, embedding_dims, nb_filter, filter_length))

    model.add(Dense(hidden_dims))
    #model.add(Dropout(drop_out_prob))
    model.add(Activation('relu'))
    model.add(Dense(nb_class))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def convLayer(inp_dim, embedding_dims, nb_filter, filter_length):
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
