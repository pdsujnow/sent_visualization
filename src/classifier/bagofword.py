import sys
from classifier import Classifier
import numpy as np
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search
from word2vec.globvemongo import Globve
import warnings

import csv

class BagofWord(Classifier):
    def init(self):
        self.globve = Globve()#wordvector model

    def un_init(self):
        del self.globve

    def text2vec(self, text):
        X = np.zeros(self.globve.dimension)
        i=0
        for word in [w.decode('utf8') for w in text]:
            if word in self.globve:
                i+=1
                X = X+self.globve[word] 
        if i>0:
            X=X/i
        return X

    def fit(self, sentences, y, n_folds=5, random_state=0):

        X = np.array([self.text2vec(s) for s in sentences])

        np.random.seed(random_state)

        svmclf = svm.LinearSVC(dual=False)
        parameters = dict(C = np.logspace(-5,1, 8))
        self.clfs = []
        for i in range(y.shape[1]):
            yy = y[:,i]
            cv = StratifiedKFold(yy, n_folds=n_folds, shuffle=True)
            clf = grid_search.GridSearchCV(svmclf, parameters, scoring='f1', n_jobs=-1, cv=cv)
            with warnings.catch_warnings(): 
                warnings.simplefilter("ignore")
                clf.fit(X, yy)
            print clf.best_params_, clf.best_score_
            self.clfs.append(clf.best_estimator_)


    def predict_prob(self, feature):
        return [clf.decision_function(feature)[0] for clf in self.clfs]
