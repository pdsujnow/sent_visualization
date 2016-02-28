#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import csv
import pickle
import numpy as np
from sklearn.svm import LinearSVC
from classifier.cnn import CNN
from model import Model
import dataloader
from feature.extractor import feature_fuse, W2VExtractor, CNNExtractor
import logging
logging.basicConfig(level=logging.DEBUG)


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '..', 'corpus')
MODEL_DIR = os.path.join(MODULE_DIR, '..', 'model')


def select_feature(feature_types):
    if len(feature_types) == 1:
        return [feature_types[0]()]
    query = 'Possible Features:\n'
    for i, ftp in enumerate(feature_types):
        query += '{}. {}\n'.format(i, ftp.__name__)
    selection = raw_input(query)
    selection = [int(f) for f in selection.split(',')]
    return [feature_types[i]() for i in selection]


if __name__ == "__main__":
    # Train classifiers

    corpus_name = 'panglee'
    corpus_dir = os.path.join(CORPUS_DIR, corpus_name)

    corpus = dataloader.load(corpus_dir)

    sentences, labels = list(corpus['sentence']), list(corpus['label'])

    feature_extractors = [CNNExtractor(mincount=0)]
    X, y = feature_fuse(feature_extractors, sentences, labels)
    clf = CNN(vocabulary_size=feature_extractors[0].vocabulary_size,
              nb_filters=100,
              embedding_dims=300,
              filter_length=[3, 4, 5],
              drop_out_prob=0.5,
              maxlen=X.shape[1],
              nb_class=len(feature_extractors[0].literal_labels))

    from sklearn.cross_validation import StratifiedKFold
    from keras.utils.np_utils import to_categorical
    cv = StratifiedKFold(y, n_folds=10, shuffle=True)
    # y = to_categorical(y)
    for train_ind, test_ind in cv:
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
        clf.fit(X_train, y_train,
                batch_size=50,
                nb_epoch=20,
                show_accuracy=True,
                validation_data=(X_test, y_test))

    # feature_extractors = [W2VExtractor()]
    # X, y = feature_fuse(feature_extractors, sentences, labels)
    # clf = LinearSVC()
    # OVO = True
    # parameters = dict(C=np.logspace(-5, 1, 8))
    # dump_file = os.path.join(MODEL_DIR, corpus_name + '_svm')
    # model = Model(clf, feature_extractors, OVO=OVO)
    # model.grid_search(X, y, parameters=parameters, n_jobs=-1)

    # feature_extractors = [CNNExtractor(mincount=2)]
    # X, y = feature_fuse(feature_extractors, sentences, labels)
    # clf = CNN(vocabulary_size=feature_extractors[0].vocabulary_size,
    #         maxlen=X.shape[1],
    #         nb_class=len(feature_extractors[0].literal_labels))
    # OVO = False
    # parameters = dict(batch_size=[32], nb_epoch=[4], verbose=[0])
    # dump_file = os.path.join(MODEL_DIR, corpus_name + '_cnn')
    # model = Model(clf, feature_extractors, OVO=OVO)
    # model.grid_search(X, y, parameters=parameters, n_jobs=1)

    #  model.dump_to_file(dump_file)
    #  for fe in feature_extractors:
    #     del fe
    # del feature_extractors

