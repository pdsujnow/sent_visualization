#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pickle
import itertools
import numpy as np
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search
from feature_extractor.preprocessor import preprocess

class Model(object):
    def __init__(self, clf, feature_extractors):
        self.clf = clf
        self.feature_extractors = feature_extractors
        self.init_too_large()

    def init_too_large(self):
        for fe in self.feature_extractors:
            fe.init_too_large()

    def del_too_large(self):
        for fe in self.feature_extractors:
            fe.del_too_large()

    @staticmethod
    def load_from_file(fname):
        with open(fname) as f:
            model = pickle.load(f)
        model.init_too_large()
        return model

    def dump_to_file(self, fname):
        self.del_too_large()
        with open(fname, 'w') as f:
            pickle.dump(self, f)

    def grid_search(self, sentences, labels, OVO=False, n_folds=5, scoring='f1',parameters=None):
        X, y, self.emotion_labels = self._prepare_training(sentences, labels)

        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore")
            clfs = []
            if OVO:
                for i in range(y.shape[1]):
                    clfs.append(self._grid_search(X, y[:,i], n_folds, scoring, parameters))
            else:
                clfs.append(self._grid_search(X, y, n_folds, scoring, parameters))
            self.clf = clfs

    def _grid_search(self, X, y, n_folds, scoring, parameters):
        cv = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
        clf = grid_search.GridSearchCV(self.clf, parameters, scoring='f1', n_jobs=-1, cv=cv)
        clf.fit(X, y)

        print clf.best_params_, clf.best_score_
        return clf.best_estimator_


    def _prepare_training(self, sentences, labels):
        emotion_labels = list(set(itertools.chain(*labels)))
        print "Labels: ", emotion_labels
        y = [[emotion_labels.index(l) for l in row] for row in labels]
        y = MultiLabelBinarizer().fit_transform(y)
        sentences = [preprocess(s) for s in sentences]

        Xs = []
        for fe in self.feature_extractors:
            fe.pre_calculate(sentences)

        X = np.array([self.text2vec(s) for s in sentences])

        return X, y, emotion_labels


    def text2vec(self, sentence):
        Xs=[]
        for fe in self.feature_extractors:
            Xs.append(fe.extract(sentence))
        return Xs[0] if len(Xs)==1 else np.concatenate(Xs, axis=1)

    def predict(self, text):
        feature = self.text2vec(preprocess(text))
        if not feature.any():
            return [0]*len(self.clf)
        
        return [c.decision_function(feature)[0] for c in self.clf]
