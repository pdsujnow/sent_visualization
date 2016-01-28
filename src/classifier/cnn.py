#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import itertools
from collections import Counter
from classifier import Classifier


from keras.preprocessing import sequence
from keras.models import Sequential, Graph
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.layers import containers

class CNN(Classifier):

    def init(self):
        if 'arch_file' not in self.__dict__:
            return
        with open(self.arch_file) as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(self.weight_file)

    def un_init(self):
        del self.model

    def dump_to_file(self, dump_file):
        self.arch_file = dump_file+'_arch.json'
        self.weight_file = dump_file+'_weights.h5'
        with open(self.arch_file, 'w') as f:
            f.write(self.model.to_json())
        self.model.save_weights(self.weight_file)
        super(CNN, self).dump_to_file(dump_file)

    def text2vec(self, text):
        if type(text[0]) is unicode:
            text = [t.encode('utf8') for t in text]
        text = self.to_given_length(text, self.maxlen)

        res = np.zeros((1,self.maxlen))
        for i, word in enumerate(text):
            if word in self.vocabulary:
                res[0][i] = self.vocabulary[word]
            else:
                res[0][i] = self.vocabulary['<PAD/>']
        return  np.array(res)

    def fit(self, sentences, y, batch_size=32, nb_epoch=4, random_state=0, validation_split=0.1,
            embedding_dims=128, nb_filter=250, filter_length=[3,4], hidden_dims=250, drop_out_prob=0.25):

        self.maxlen = max(len(x) for x in sentences)
        sentences = [self.to_given_length(s, self.maxlen) for s in sentences]
        self.vocabulary = self.build_vocab(sentences)

        X = np.concatenate([self.text2vec(s) for s in sentences])
        nb_class = y.shape[1]

        np.random.seed(random_state)

        self.model = Sequential()
        self.model.add(Masking(mask_value=0))
        self.model.add(Embedding(len(self.vocabulary), embedding_dims, input_length=self.maxlen))

        #self.model.add(LSTM(embedding_dims))
        #self.model.add(Dropout(0.5))
        #self.model.add(Dense(hidden_dims))
        #self.model.add(Dropout(drop_out_prob))
        #self.model.add(Activation('relu'))
        #self.model.add(Dense(y.shape[1]))
        #self.model.add(Activation('sigmoid'))

        self.model.add(Dropout(drop_out_prob))
        self.model.add(self.convLayer(self.maxlen, embedding_dims, nb_filter, filter_length))
        self.model.add(Dense(hidden_dims))
        self.model.add(Dropout(drop_out_prob))
        self.model.add(Activation('relu'))
        self.model.add(Dense(y.shape[1]))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        self.model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, validation_split=validation_split)


    def predict_prob(self, feature):
        return self.model.predict(feature).tolist()


    def convLayer(self, inp_dim, embedding_dims, nb_filter, filter_length=[2]):
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



    def to_given_length(self, sentence, length, padding_word="<PAD/>"):
        sentence = sentence[:length]
        return sentence + [padding_word] * (length - len(sentence))


    def build_vocab(self, sentences, mincount=5):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1]>mincount]
        assert vocabulary_inv[0]=='<PAD/>'
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return vocabulary

