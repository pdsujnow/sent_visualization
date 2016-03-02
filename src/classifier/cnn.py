#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from scikit_learn import KerasClassifier
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential, Graph
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D
from keras.layers import containers
# import tensorflow as tf
# from keras import backend as K
# from keras.backend import tensorflow_backend as KTF
#
#
# class MaskEmbedding(Embedding):
#
#     def __init__(self, input_dim, output_dim, use_mask=True, **kwargs):
#         self.use_mask = use_mask
#         super(MaskEmbedding, self).__init__(input_dim, output_dim, **kwargs)
#
#     def get_output(self, train=False):
#         X = self.get_input(train)
#
#         if self.use_mask:
#             m = np.ones((self.input_dim, self.output_dim))
#             m[0] = [0] * self.output_dim
#             mask = tf.constant(m, dtype=self.W.dtype)
#             outW = K.gather(self.W, X)
#             outM = K.gather(mask, X)
#             return outW * outM
#         else:
#             return K.gather(self.W, X)

# MyEmbedding = MaskEmbedding


class TwoD_CNN(KerasClassifier):

    def __call__(self,
                vocabulary_size=5000,
                maxlen=100,
                embedding_dims=100,
                nb_filter=250,
                filter_length=[3],
                hidden_dims=250,
                nb_class=2,
                drop_out_prob=0.25):

        model = Sequential()

        model.add(Embedding(vocabulary_size, embedding_dims, input_length=maxlen))
        model.add(Reshape((1, maxlen, embedding_dims)))

        model.add(self.convLayer2d(maxlen, embedding_dims, nb_filter, filter_length))

        model.add(Dense(hidden_dims))
        model.add(Activation('relu'))
        model.add(Dropout(drop_out_prob))

        assert(nb_class > 1)
        if nb_class == 2:
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            model.compile(optimizer='adadelta', loss='binary_crossentropy',
                          class_mode='binary')
        else:
            model.add(Dense(nb_class))
            model.add(Activation('softmax'))
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                          class_mode='categorical')

        return model

    def convLayer2d(self, img_row, img_col, nb_filter, filter_length):
        c = containers.Graph()
        input_shape = (1, img_row, img_col)
        c.add_input(name='input', input_shape=input_shape)
        inps = []
        for i in filter_length:
            c.add_node(containers.Sequential([Convolution2D(nb_filter=nb_filter,
                                                 nb_row=i,
                                                 nb_col=i,
                                                 border_mode='valid',
                                                 activation='relu',
                                                 dim_ordering='th',
                                                 subsample=(1, i),
                                                 input_shape=input_shape,
                                                            ),
                                            MaxPooling2D(pool_size=(2, 2)),
                                            Flatten()]),
                       name='Conv{}'.format(i), input='input')
            inps.append('Conv{}'.format(i))

        if len(inps) == 1:
            c.add_output('output', input=inps[0])
        else:
            c.add_output('output', inputs=inps)

        return c


class OneD_CNN(KerasClassifier):

    def __call__(self,
                vocabulary_size=5000,
                maxlen=100,
                embedding_dims=100,
                nb_filter=250,
                filter_length=[3],
                hidden_dims=250,
                nb_class=2,
                drop_out_prob=0.25, embedding_weights=None):

        model = Sequential()

        if embedding_weights is not None:
            for i in range(embedding_weights.shape[0]):
                if not embedding_weights[i].any():
                    embedding_weights[i] = np.random.uniform(low=-0.05, high=0.05, size=(embedding_dims,))
        model.add(Embedding(vocabulary_size, embedding_dims, input_length=maxlen, weights=[embedding_weights]))
        # model.add(Dropout(drop_out_prob))

        model.add(self.convLayer(maxlen, embedding_dims,
                                 nb_filter, filter_length))

        model.add(Dense(hidden_dims))
        model.add(Activation('relu'))
        model.add(Dropout(drop_out_prob))

        assert(nb_class > 1)
        if nb_class == 2:
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            model.compile(optimizer='adadelta', loss='binary_crossentropy',
                          class_mode='binary')
        else:
            model.add(Dense(nb_class))
            model.add(Activation('softmax'))
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                          class_mode='categorical')

        return model

    def convLayer(self, img_row, img_col, nb_filter, filter_length):
        c = containers.Graph()
        input_shape = (img_row, img_col)
        c.add_input(name='input', input_shape=input_shape)
        inps = []
        for i in filter_length:
            c.add_node(containers.Sequential([Convolution1D(nb_filter=nb_filter,
                                                 filter_length=i,
                                                 border_mode='valid',
                                                 activation='relu',
                                                 subsample_length=1,
                                                 input_shape=input_shape,
                                                            ),
                                            MaxPooling1D(pool_length=2),
                                            Flatten()]),
                       name='Conv{}'.format(i), input='input')
            inps.append('Conv{}'.format(i))

        if len(inps) == 1:
            c.add_output('output', input=inps[0])
        else:
            c.add_output('output', inputs=inps)

        return c

    def post_load(self):
        with open(self.arch_file) as f:
            if type(MyEmbedding) is not Embedding:
                self.model = model_from_json(f.read(), {"MyEmbedding": MyEmbedding})
            else:
                self.model = model_from_json(f.read())

        self.model.load_weights(self.weight_file)

    def pre_dump(self, dump_file):
        self.arch_file = dump_file + '_arch.json'
        self.weight_file = dump_file + '_weights.h5'

        with open(self.arch_file, 'w') as f:
            f.write(self.model.to_json())
        self.model.save_weights(self.weight_file, overwrite=True)
        del self.model
