#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
from scikit_learn import KerasClassifier
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential, Graph
from keras.models import model_from_json
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, ZeroPadding1D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute, RepeatVector, Layer
from keras.layers.recurrent import LSTM
from keras.layers import containers
import theano
from keras import backend as K

import logging
logging.basicConfig(level=logging.DEBUG)

class MyEmbedding(Embedding):
    def __init__(self, input_dim, output_dim, use_mask=True, **kwargs):
        self.use_mask = use_mask
        super(MyEmbedding, self).__init__(input_dim, output_dim, **kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.use_mask:
            m = np.ones((self.input_dim, self.output_dim),
                        dtype=theano.config.floatX)
            m[0] = [0] * self.output_dim
            mask = theano.tensor.constant(m, dtype=theano.config.floatX)
            outW = K.gather(self.W, X)
            outM = K.gather(mask, X)
            return outW * outM
        else:
            return K.gather(self.W, X)


class CNNS(KerasClassifier):
    def add_embedding(self, model, vocabulary_size, embedding_dims, maxlen, use_my_embedding, embedding_weights=None):

        if embedding_weights is not None:
            for i in range(embedding_weights.shape[0]):
                if not embedding_weights[i].any():
                    embedding_weights[i] = np.random.uniform(
                        low=-0.05, high=0.05, size=(embedding_dims,))
            if use_my_embedding:
                model.add(MyEmbedding(vocabulary_size, embedding_dims,
                                      input_length=maxlen, weights=[embedding_weights]))
            else:
                model.add(Embedding(vocabulary_size, embedding_dims,
                                    input_length=maxlen, weights=[embedding_weights]))
        else:
            if use_my_embedding:
                model.add(MyEmbedding(vocabulary_size,
                                      embedding_dims, input_length=maxlen))
            else:
                model.add(Embedding(vocabulary_size,
                                    embedding_dims, input_length=maxlen))

        padding_len = self.calc_padding_len(maxlen)
        if padding_len>0:
            model.add(ZeroPadding1D(padding_len))

        return maxlen+2*padding_len

    def log_params(self, params):
        weights = params.pop('embedding_weights')
        if weights is not None:
            params.update({'embedding_weights': 'given'})
        else:
            params.update({'embedding_weights': 'random'})
        print params

    def add_full(self, model, hidden_dims, drop_out_prob, nb_class,):
        model.add(Flatten())
        model.add(Dense(hidden_dims))
        model.add(Activation('relu'))
        model.add(Dropout(drop_out_prob))
        # model.add(LSTM(70))

        assert(nb_class > 1)
        if nb_class == 2:
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            model.compile(optimizer='adadelta', loss='binary_crossentropy',
                          class_mode='binary')
        else:
            model.add(Dense(nb_class))
            model.add(Activation('softmax'))
            print 'begin compile..'
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                          class_mode='categorical')
            print 'end compile'

    def post_load(self):
        with open(self.arch_file) as f:
            self.model = model_from_json(
                f.read(), {"MyEmbedding": MyEmbedding})
        self.model.load_weights(self.weight_file)

    def pre_dump(self, dump_file):
        self.arch_file = dump_file + '_arch.json'
        self.weight_file = dump_file + '_weights.h5'

        with open(self.arch_file, 'w') as f:
            f.write(self.model.to_json())
        self.model.save_weights(self.weight_file, overwrite=True)
        del self.model

    def calc_padding_len(self, ori_len):
        assert ori_len % 2 ==0
        align = [2**i for i in range(10)]
        if ori_len not in align:
            for l in align:
                if l>ori_len:
                    return (l-ori_len)/2

class RNN(CNNS):

    def __call__(self,
                 vocabulary_size=5000,
                 embedding_dims=300,
                 hidden_dims=250,
                 nb_class=2,
                 drop_out_prob=0.5,
                 use_my_embedding=False,
                 embedding_weights=None):

        self.log_params(locals())
        model = Sequential()
        maxlen = self.add_embedding(model, vocabulary_size, embedding_dims,maxlen,
                                    use_my_embedding, embedding_weights)

        self.add_full(model, hidden_dims, drop_out_prob, nb_class)
        return model

class Multi_OneD_CNN(CNNS):
    def __call__(self,
                 vocabulary_size=5000,
                 maxlen=100,
                 embedding_dims=300,
                 nb_filter=300,
                 filter_length=[3],
                 layer=-1,
                 hidden_dims=250,
                 nb_class=2,
                 drop_out_prob=0.5,
                 use_my_embedding=False,
                 embedding_weights=None):

        filter_row = filter_length=filter_length[0]
        filter_col = embedding_dims
        nb_filter = embedding_dims
        self.log_params(locals())
        model = Sequential()
        maxlen = self.add_embedding(model, vocabulary_size, embedding_dims,maxlen,
                                    use_my_embedding, embedding_weights)

        max_possible_layer = int(math.log(maxlen, 2))-1
        if layer>0:
            assert layer < max_possible_layer
        else:
            layer = max_possible_layer

        for i in range(layer):
            # model.add(Dropout(drop_out_prob/(i+1)))
            model.add(Convolution1D(nb_filter=nb_filter,
                                    filter_length=filter_row,
                                    border_mode='same',
                                    activation='relu',
                                    subsample_length=1))
            model.add(MaxPooling1D(2))

            # If max_possible_layer is adopted, add one more layer
            # if i == max_possible_layer-1:
            #     logging.debug("Last layer added")
            #     model.add(Convolution1D(nb_filter=nb_filter,
            #                             filter_length=2,
            #                             border_mode='valid',
            #                             activation='relu',
            #                             subsample_length=1))

            if i == max_possible_layer-1:
                logging.debug("Last layer added")
                model.add(Reshape(1, 2, embedding_dims))
                model.add(Convolution2D(nb_filter=nb_filter,
                                        nb_row=2,
                                        nb_col=1,
                                        border_mode='valid',
                                        activation='relu',
                                        dim_ordering='th',
                                        subsample=(1, 1)))

        self.add_full(model, hidden_dims, drop_out_prob, nb_class)
        return model

class MyCNN(CNNS):
    def __call__(self,
                 vocabulary_size=5000,
                 maxlen=100,
                 embedding_dims=300,
                 filter_length=[3],
                 layer=-1,
                 hidden_dims=250,
                 nb_class=2,
                 drop_out_prob=0.5,
                 use_my_embedding=False,
                 embedding_weights=None):

        filter_row = filter_length[0]
        filter_col = 1
        nb_filter = embedding_dims

        self.log_params(locals())
        model = Sequential()
        maxlen = self.add_embedding(model, vocabulary_size, embedding_dims,maxlen,
                                    use_my_embedding, embedding_weights)

        model.add(Permute((2, 1)))
        model.add(Reshape((nb_filter* maxlen,)))
        model.add(RepeatVector(filter_col))
        model.add(Permute((2, 1)))
        model.add(Reshape((nb_filter, maxlen, filter_col)))

        max_possible_layer = int(math.log(maxlen, 2))-1
        if layer>0:
            assert layer < max_possible_layer 
        else:
            layer = max_possible_layer

        for i in range(layer):
            # model.add(Dropout(drop_out_prob/(i+1)))
            model.add(Convolution2D(nb_filter=nb_filter,
                                    nb_row=filter_row,
                                    nb_col=1,
                                    border_mode='same',
                                    activation='relu',
                                    dim_ordering='th',
                                    subsample=(1, 1)))
            model.add(MaxPooling2D(pool_size=(2, 1)))

            # If max_possible_layer is adopted, add one more layer
            if i == max_possible_layer-1:
                logging.debug("Last layer added")
                model.add(Convolution2D(nb_filter=nb_filter,
                                        nb_row=2,
                                        nb_col=1,
                                        border_mode='valid',
                                        activation='relu',
                                        dim_ordering='th',
                                        subsample=(1, 1)))

        self.add_full(model, hidden_dims, drop_out_prob, nb_class)
        return model

class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)

class MyRegularCNN(CNNS):
    def graphadd(self, block, node, input_name, name):
        block.add_node(node, name=name, input=input_name)
        return name

    def __call__(self,
                 vocabulary_size=5000,
                 maxlen=100,
                 embedding_dims=300,
                 filter_length=[3],
                 layer=-1,
                 skip=2,
                 hidden_dims=250,
                 nb_class=2,
                 drop_out_prob=0.5,
                 use_my_embedding=False,
                 embedding_weights=None):

        filter_row = filter_length[0]
        filter_col = 1
        nb_filter = embedding_dims

        self.log_params(locals())
        model = Sequential()
        maxlen = self.add_embedding(model, vocabulary_size, embedding_dims,maxlen,
                                    use_my_embedding, embedding_weights)

        model.add(Permute((2, 1)))
        model.add(Reshape((nb_filter* maxlen,)))
        model.add(RepeatVector(filter_col))
        model.add(Permute((2, 1)))
        model.add(Reshape((nb_filter, maxlen, filter_col)))

        max_possible_layer = int(math.log(maxlen, 2))-1
        if layer>0:
            assert layer < max_possible_layer
        else:
            layer = max_possible_layer

        for i in range(layer):
            input_shape = (nb_filter, maxlen/2**i, filter_col)
            block = containers.Graph()
            input_name = 'block_{}_input'.format(i)
            identity_name = 'block_{}_identity'.format(i)
            block.add_input(input_name, input_shape=input_shape)
            self.graphadd(block, Identity(), input_name, identity_name)

            prev_output = input_name
            for j in range(skip):
                conv = Convolution2D(nb_filter=nb_filter,
                                     nb_row=filter_row,
                                     nb_col=1,
                                     border_mode='same',
                                     activation='relu',
                                     dim_ordering='th',
                                     subsample=(1, 1))
                prev_output = self.graphadd(block, conv, prev_output, 'conv_{}_{}'.format(i,j))
                if j < skip:
                    prev_output = self.graphadd(block, Dropout(drop_out_prob), prev_output, 'dropout_{}_{}'.format(i,j))

            block.add_output('block_{}_output'.format(i), inputs=[prev_output, identity_name], merge_mode='sum')

            model.add(block)
            model.add(MaxPooling2D(pool_size=(2, 1)))

            if i == max_possible_layer-1:
                logging.debug("Last layer added")
                model.add(Convolution2D(nb_filter=nb_filter,
                                        nb_row=2,
                                        nb_col=1,
                                        border_mode='valid',
                                        activation='relu',
                                        dim_ordering='th',
                                        subsample=(1, 1)))

        self.add_full(model, hidden_dims, drop_out_prob, nb_class)
        return model
