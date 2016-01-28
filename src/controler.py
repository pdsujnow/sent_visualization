#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import csv
from preprocessor import preprocess
from classifier.classifier import Classifier

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '../corpus/')
TRAINED_DIR = os.path.join(MODULE_DIR, '../model/')

def load_data_and_labels(file_name, exclude_labels=['irrelevant'], col_label_name='Sentiment', col_sentence_name='sentence'):
    with open(file_name) as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

    sentences = []
    labels = []

    for row in data:
        e = row[col_label_name]
        if e in exclude_labels:
            continue
        sentences.append(preprocess(row[col_sentence_name]))
        labels.append([e])

    return [sentences, labels]


def install_all_model(models, dirname, fnames):
    if dirname=='.':
        return
    for fname in fnames:
        path = os.path.join(dirname,fname)
        if '.' in os.path.basename(path):
            continue
        if os.path.isfile(path):
            clf = Classifier().load_from_file(path)
            models[clf.name] = clf

class Controler():
    def __init__(self): 
        self.models={}
        root_dir = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))
        self.TRAINED_DIR = os.path.join(root_dir, TRAINED_DIR)
        print "Loading models..."
        os.path.walk(self.TRAINED_DIR, install_all_model, self.models)
        print "All models loaded"

    def list_model(self):
        return {m.name: m.emotions for m in self.models.values()}

    def predict(self, model_name, text):
        return self.models[model_name].predict(preprocess(text))

if __name__ == "__main__": 
    #Train classifiers

    from classifier.bagofword import BagofWord
    from classifier.cnn import CNN
    corpus = 'sanders'

    corpus_f = CORPUS_DIR+corpus+'/parsed.csv'
    sentences, labels = load_data_and_labels(corpus_f)
    clf = CNN()
    clf.train(sentences, labels, random_state=0) 
    clf.dump_to_file(TRAINED_DIR+corpus+'_cnn')

    #corpus_f = CORPUS_DIR+corpus+'/parsed.csv'
    #sentences, labels = load_data_and_labels(corpus_f)
    #clf = BagofWord()
    #clf.train(sentences, labels, n_folds=5, random_state=0) 
    #clf.dump_to_file(TRAINED_DIR+corpus+'_bow')

