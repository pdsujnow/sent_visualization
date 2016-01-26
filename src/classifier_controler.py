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

    def __init__(self):
        self.init()

    def load_from_file(self, file_name):

        assert os.path.isfile(file_name), "{} not found".format(file_name)
        with open(file_name) as f:
            clf = pickle.load(f)
        clf.init() #Some variables might be too big and not dumped into the file, initial those variables in this functtion

        return clf

    def dump_to_file(self, dump_file):
        self.un_init() #delete varaibles initialized in 'init()' function
        self.name = os.path.basename(dump_file)
        with open(dump_file, 'w') as f:
            pickle.dump(self, f)

    def predict(self, text):
        feature = self.text2vec(preprocess(text))
        return [self.predict_prob(clf, feature)[0] for clf in self.child_clfs]


    def train(self, corpus, dump_file, **kwargs):
        sentences, y, emotions = load_data_and_labels(corpus)
        child_clfs = self.fit(sentences, y, **kwargs)
        self.emotions = emotions
        self.child_clfs = child_clfs

        self.dump_to_file(dump_file)

    def init(self):
        """
        If the sub class has some variables that is too large and do not want to be dumped.
        Override this function and initialize those variables in it 
        Also, delete those variables in the 'un_init' function to prevent them from being dumpped. 
        """
        pass

    def un_init(self):
        """Please refer to init"""
        pass

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
            clf = Classifier().load_from_file(path)
            models[clf.name] = clf

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
        if len(self.models)==0:
            self.list_model()
        return self.models[model_name].predict(text)

if __name__ == "__main__": 
    from classifier.bagofword import BagofWord
    #Train classifiers
    corpus='sanders'
    clf = BagofWord()
    clf.train(corpus=CORPUS_DIR+corpus+'/parsed.csv',
                dump_file=TRAINED_DIR+corpus+'_bow',
                n_folds=5, random_state=0) 
