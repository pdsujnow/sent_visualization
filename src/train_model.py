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
from feature_extractor import feature_fuse, W2VExtractor, CNNExtractor

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '..', 'corpus')
MODEL_DIR = os.path.join(MODULE_DIR, '..', 'model')

def load_data_and_labels(file_name, exclude_labels=['irrelevant']):
    with open(file_name) as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

    sentences = []
    labels = []

    for row in data:
        e = row['sentiment']
        if e in exclude_labels:
            continue
        sentences.append(row['sentence'])
        labels.append([e])

    return [sentences, labels]

def select_feature(feature_types):
    if len(feature_types)==1:
        return [feature_types[0]()]
    query = 'Possible Features:\n'
    for i, ftp in enumerate(feature_types):
        query += '{}. {}\n'.format(i, ftp.__name__)
    selection = raw_input(query)
    selection = [int(f) for f in selection.split(',')]
    return [feature_types[i]() for i in selection]


if __name__ == "__main__": 
    #Train classifiers

    corpus = 'sanders'
    corpus_f = os.path.join(CORPUS_DIR, corpus, 'parsed.csv')
    sentences, labels = load_data_and_labels(corpus_f)

    #feature_extractors = [W2VExtractor()]
    #X, y = feature_fuse(feature_extractors, sentences, labels)
    #clf = LinearSVC()
    #OVO = True 
    #parameters = dict(C = np.logspace(-5,1, 8))
    #dump_file = os.path.join(MODEL_DIR, corpus+'_svm')

    feature_extractors = [CNNExtractor()]
    X, y = feature_fuse(feature_extractors, sentences, labels)
    clf = CNN( vocabulary_size = feature_extractors[0].vocabulary_size,
            maxlen=X.shape[1],
            nb_class=len(feature_extractors[0].literal_labels))
    OVO = False
    parameters = dict( filter_length = [[2,3,4]], nb_epoch=[1], verbose=[0])
    dump_file = os.path.join(MODEL_DIR, corpus+'_cnn')


    model = Model(clf, feature_extractors, OVO=OVO)
    model.grid_search(X, y,n_folds=2, parameters=parameters)
    model.dump_to_file(dump_file)

    for fe in feature_extractors:
        del fe
    del feature_extractors


    #corpus = 'LJ40k'
    #corpus_f = CORPUS_DIR+corpus+'/parsed.csv'
    #sentences, labels = load_data_and_labels(corpus_f, col_label_name='sentiment')

    #corpus = 'LJ40k'
    #corpus_f = CORPUS_DIR+corpus+'/reduced_parsed.csv'
    #sentences, labels = load_data_and_labels(corpus_f, col_label_name='sentiment')
