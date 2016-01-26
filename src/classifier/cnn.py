#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import itertools
from collections import Counter
from classifier import Classifier


from keras.preprocessing import sequence
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.layers import containers

class CNN(Classifier):

    def text2vec(self, text):
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

        X = np.zeros(self.globve.dimension)
        for word in [w.decode('utf8') for w in text]:
            if word in self.globve:
                X = X+self.globve[word] 
        return X


    def fit(self, sentences, y, batch_size=32, nb_epoch=4, random_state=0):
        sentences_padded = pad_sentences(sentences)
        print (sentences_padded)
        sys.exit(-1)
        vocabulary, vocabulary_inv = self.build_vocab(sentences_padded)


        self.maxlen = maxlen
        self.vocabulary = vocabulary
        self.vocabulary_inv = vocabulary_inv

        X = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
        nb_class = y.shape[1]

        # set parameters:
        vocab_size = len(vocabulary)
        maxlen = len(sentences_padded[0])

        embedding_dims = 100
        nb_filter = 250
        filter_length = [3,4]
        hidden_dims = 250
        drop_out_prob = 0.25

        np.random.seed(random_state)

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dims, input_length=maxlen))
        model.add(Dropout(drop_out_prob))
        model.add(convLayer(filter_length))
        model.add(Dense(hidden_dims))
        model.add(Dropout(drop_out_prob))
        model.add(Activation('relu'))
        model.add(Dense(num_class))
        model.add(Activation('sigmoid'))

        model.compile(loss='crossentropy',
                     optimizer='rmsprop')

        model.fit(X, y, batch_size=batch_size,
                  nb_epoch=nb_epoch, show_accuracy=True,
                  validation_data=(X_test, y_test))

        return [model]

    def predict_prob(self, clf, feature):
        return clf.predict_prob(feature)

    def pad_sentences(self, sentences, padding_word="<PAD/>"):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        sequence_length = max(len(x) for x in sentences)
        padded_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        return padded_sentences


    def build_vocab(sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def convLayer(self, filter_length=[2]):
        c = containers.Graph()
        c.add_input(name='input', input_shape=(maxlen, embedding_dims))
        for i in filter_length:
            c.add_node(Convolution1D(nb_filter=nb_filter, filter_length=i, border_mode='valid', activation='relu',subsample_length=1),
                                    name='_Conv{}'.format(i), input='input')
            c.add_node(containers.Sequential([
                                MaxPooling1D(pool_length=2),
                                Flatten()]),
                                name='Conv{}'.format(i), input='_Conv{}'.format(i))
        inps = ['Conv{}'.format(i) for i in filter_length]
        if len(inps)==1:
            c.add_output('output', input=inps[0])
        else:
            c.add_output('output', inputs=inps)

        return c

