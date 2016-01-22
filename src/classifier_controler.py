#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pickle
import csv
from sklearn.preprocessing import MultiLabelBinarizer
from data_helpers import load_data_and_labels, preprocess

CORPUS_DIR = '../corpus/'
TRAINED_DIR = '../model/'


class Classifier(object):
    def __init__(self, file_name=None):
        self.file_name =file_name
        self.name = os.path.basename(file_name)

        if os.path.isfile(file_name):
            with open(file_name) as f:
                model = pickle.load(f)
            self.emotions = model['emotions']
            self.clfs = model['clfs']


    def predict(self, text):
        feature = self.text2vec(preprocess(text))
        return [self.predict_prob(clf, feature)[0] for clf in self.clfs]

    def train(self, corpus, **kwargs):
        sentences, y, emotions = load_data_and_labels(corpus)
        clfs = self.fit(sentences, y, **kwargs)
        model = {'emotions': emotions, 'clfs': clfs, 'type':type(self)}
        with open(self.file_name, 'w') as f:
            pickle.dump(model, f)

    def fit(self, sentences, y, **kwargs):
        raise NotImplementedError 

    def text2vec(self, text):
        raise NotImplementedError 

    def predict_prob(self, clf, feature):
        raise NotImplementedError 


def install_all_model(models, dirname, fnames):
    if dirname=='.':
        return
    for fname in fnames:
        path = os.path.join(dirname,fname)
        if os.path.isfile(path):
            with open(path) as f:
                model = pickle.load(f)
            c = model['type'](path)
            models[c.name] = c

class ClassifierControler():
    def __init__(self): 
        self.models={}
        root_dir = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))
        self.TRAINED_DIR = os.path.join(root_dir, TRAINED_DIR)

    def list_model(self):
        if len(self.models)==0:
            os.path.walk(self.TRAINED_DIR, install_all_model, self.models)
        return {m.name: m.emotions for m in self.models.values()}

    def predict(self, model_name, text):
        return self.models[model_name].predict(text)

if __name__ == "__main__": 
    from classifier.bagofword import BagofWord
    #Train classifiers
    corpus='sanders'
    clf = BagofWord(TRAINED_DIR+corpus+'_bow')
    clf.train(CORPUS_DIR+corpus+'/parsed.csv', n_folds=5, random_state=0) 
