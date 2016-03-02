#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import csv
import cPickle
import numpy as np
from sklearn.svm import LinearSVC
from classifier.cnn import OneD_CNN, TwoD_CNN
from model import Model
from word2vec.word2vec import Word2Vec
import dataloader
from feature.extractor import feature_fuse, W2VExtractor, CNNExtractor
import logging
logging.basicConfig(level=logging.DEBUG)


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '..', 'corpus')
MODEL_DIR = os.path.join(MODULE_DIR, '..', 'model')
CACHE_DIR = os.path.join(MODULE_DIR, '..', 'cache')


def select_feature(feature_types):
    if len(feature_types) == 1:
        return [feature_types[0]()]
    query = 'Possible Features:\n'
    for i, ftp in enumerate(feature_types):
        query += '{}. {}\n'.format(i, ftp.__name__)
    selection = raw_input(query)
    selection = [int(f) for f in selection.split(',')]
    return [feature_types[i]() for i in selection]


def load_embedding(vocabulary, filename):
    filename = os.path.join(CACHE_DIR, filename)
    if os.path.isfile(filename):
        with open(filename) as f:
            W = cPickle.load(f)
        return W
    else:
        res = np.zeros((len(vocabulary), 300))
        w2v = Word2Vec()
        for word in vocabulary.keys():
            if word in w2v:
                ind = vocabulary[word]
                res[ind] = w2v[word]
        with open(filename, 'w') as f:
            cPickle.dump(res, f)
        return res

from sklearn.externals.joblib import Parallel, delayed
from multiprocessing import cpu_count


def reorder_embedding(w2v, lexicon_fname):
    import pandas as pd
    lexicon_fname = '../lexicon/connotation/connotation.csv'
    df = pd.read_csv(lexicon_fname, header=None)
    df.columns = ['word', 'sent']

    def get_emb(df, sent):
        words = list(df['word'][df['sent'] == sent])
        emb = []
        for word in words:
            if word in w2v:
                emb.append(w2v[word])
        return np.array(emb)

    pos_mat = get_emb(df, 'positive')
    neg_mat = get_emb(df, 'negative')

    return newInd

    # newInd = reorder_embedding(W, pos)

if __name__ == "__main__":
    # Train classifiers

    corpus_name = 'panglee'
    corpus_dir = os.path.join(CORPUS_DIR, corpus_name)

    corpus = dataloader.load(corpus_dir)
    sentences, labels = list(corpus['sentence']), list(corpus['label'])

    # from sklearn.cross_validation import StratifiedKFold
    # from keras.utils.np_utils import to_categorical
    #
    # cnn_extractor = CNNExtractor(mincount=0)
    # feature_extractors = [cnn_extractor]
    # X, y = feature_fuse(feature_extractors, sentences, labels)
    # W = load_embedding(cnn_extractor.vocabulary, corpus_name + '_emb.pkl')
    #
    # clf = OneD_CNN(vocabulary_size=cnn_extractor.vocabulary_size,
    #                nb_filters=100,
    #                embedding_dims=300,
    #                filter_length=[3],
    #                drop_out_prob=0.5,
    #                maxlen=X.shape[1],
    #                nb_class=len(cnn_extractor.literal_labels),
    #                embedding_weights=W)
    #
    # cv = StratifiedKFold(y, n_folds=10, shuffle=True)
    # for train_ind, test_ind in cv:
    #     X_train, X_test = X[train_ind], X[test_ind]
    #     y_train, y_test = y[train_ind], y[test_ind]
    #     clf.fit(X_train, y_train,
    #             batch_size=50,
    #             nb_epoch=20,
    #             show_accuracy=True,
    #             validation_data=(X_test, y_test))

    feature_extractors = [W2VExtractor()]
    X, y = feature_fuse(feature_extractors, sentences, labels)
    clf = LinearSVC()
    OVO = True
    parameters = dict(C=np.logspace(-5, 1, 8))
    dump_file = os.path.join(MODEL_DIR, corpus_name + '_svm')
    model = Model(clf, feature_extractors, OVO=OVO)
    model.grid_search(X, y, parameters=parameters, n_jobs=-1)

    # feature_extractors = [CNNExtractor(mincount=0)]
    # X, y = feature_fuse(feature_extractors, sentences, labels)
    # clf = OneD_CNN(vocabulary_size=cnn_extractor.vocabulary_size,
    #           nb_filters=100,
    #           embedding_dims=300,
    #           drop_out_prob=0.5,
    #           maxlen=X.shape[1],
    #           nb_class=len(cnn_extractor.literal_labels))
    # OVO = False
    # parameters = dict(batch_size=[50], nb_epoch=[20], show_accuracy=[True])
    # dump_file = os.path.join(MODEL_DIR, corpus_name + '_cnn')
    # model = Model(clf, feature_extractors, OVO=OVO)
    # model.grid_search(X, y, parameters=parameters, n_jobs=1)

    model.dump_to_file(dump_file)
    for fe in feature_extractors:
        del fe
    del feature_extractors
