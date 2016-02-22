#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import csv
import numpy as np
from sklearn.svm import LinearSVC
from model import Model
from feature_extractor import W2VExtractor, CNNExtractor

CORPUS_DIR = os.path.join('..', 'corpus')
MODEL_DIR = os.path.join('..', 'model')
SVM_FEATURE_EXTRACTOR = [W2VExtractor]
CNN_FEATURE_EXTRACTOR = [CNNExtractor]

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

    #from classifier.cnn import CNN

    #corpus = 'LJ40k'
    #corpus_f = CORPUS_DIR+corpus+'/parsed.csv'
    #sentences, labels = load_data_and_labels(corpus_f, col_label_name='sentiment')

    #corpus = 'LJ40k'
    #corpus_f = CORPUS_DIR+corpus+'/reduced_parsed.csv'
    #sentences, labels = load_data_and_labels(corpus_f, col_label_name='sentiment')


    corpus = 'sanders'
    corpus_f = os.path.join(CORPUS_DIR, corpus, 'parsed.csv')
    sentences, labels = load_data_and_labels(corpus_f)

    #clf = LinearSVC()
    #feature_extractors = select_feature(SVM_FEATURE_EXTRACTOR)
    #parameters = dict(C = np.logspace(-5,1, 8))
    #OVO = True #one_vs_one
    #dump_file = os.path.join(MODEL_DIR, corpus+'_svm')

    clf = CNN()
    feature_extractors = select_feature(CNN_FEATURE_EXTRACTOR)
    parameters = dict()
    OVO = False
    dump_file = os.path.join(MODEL_DIR, corpus+'_cnn')

    model = Model(clf, feature_extractors)
    model.grid_search(sentences, labels, OVO=OVO, parameters=parameters)

    model.dump_to_file(dump_file)
