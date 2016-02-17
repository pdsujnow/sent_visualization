import sys
from classifier import Classifier
import numpy as np
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search
from word2vec.globvemongo import Globve
from word2vec.word2vec import Word2Vec
from collections import Counter
import itertools

import warnings

import csv

class FeatureExtractor(object):

    def extract(self, text):
        raise NotImplementedError
    def pre_calculate(self, sentences):
        pass

class W2VExtractor(FeatureExtractor):
    def __init__(self):
        #self.model = Globve()#wordvector model
        self.model = Word2Vec()#wordvector model

    def extract(self, text):
        X = np.zeros(300)
        i=0
        for word in [w.decode('utf8') for w in text]:
            if word in self.model:
                i+=1
                X = X+self.model[word] 
        if i>0:
            X=X/i
        if float(i)/len(text)<0.6:
            X = np.zeros_like(X)

        return X

class TfIdfExtractor(FeatureExtractor):

    def pre_calculate(self, sentences):
        vocabulary = self.build_vocab(sentences)
        word_doc_mat = np.zeros((len(vocabulary), len(sentences)))
        for d, sent in enumerate(sentences):
            for word in sent:
                if word in vocabulary:
                    word_doc_mat[vocabulary[word], d]+=1
        
        self.F = [0]*len(vocabulary)
        for t, w in enumerate(vocabulary):

        self.d =0


    def extract(self, text):

        self.d+=1

    def build_vocab(self, sentences, mincount=10):
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1]>mincount]
        # Mapping from word to index
        return {x: i for i, x in enumerate(vocabulary_inv)}



FEATURE = [W2VExtractor, TfIdfExtractor]

class BagofWord(Classifier):
    def init(self, from_file=False):
        if not from_file:
            query = 'Possible Features:\n'
            for i, f in enumerate(FEATURE):
                query += '{}. {}\n'.format(i, f.__name__)
            feature_list = raw_input(query)
            feature_list = [int(f) for f in feature_list.split(',')]
            self.feature_extractor_type = [FEATURE[f] for f in feature_list]
            self.feature_extractor = [f() for f in self.feature_extractor_type]
        else:
            self.feature_extractor = [f() for f in self.feature_extractor_type]



    def un_init(self):
        for f in self.feature_extractor:
            del f
        del self.feature_extractor
            

    def text2vec(self, text):
        Xs=[]
        for f in self.feature_extractor:
            Xs.append(f.extract(text))
        if len(Xs)>1:
            return np.concatenate(Xs)
        else:
            return Xs[0]


    def fit(self, sentences, y, n_folds=5, random_state=0):

        for f in self.feature_extractor:
            f.pre_calculate(sentences)

        X = np.array([self.text2vec(s) for s in sentences])

        np.random.seed(random_state)

        svmclf = svm.LinearSVC(dual=False)
        parameters = dict(C = np.logspace(-5,1, 8))
        self.clfs = []
        for i in range(y.shape[1]):
            yy = y[:,i]

            yy1 = yy[yy==1]
            yy0 = yy[yy==0]
            XX1 = X[yy==1]
            XX0 = X[yy==0]
            index =np.random.permutation(len(yy0))[:len(yy1)]
            yy0, XX0 = yy0[index], XX0[index]
            yy=np.concatenate((yy1, yy0))
            XX = np.concatenate((XX1, XX0))

            #XX=X

            cv = StratifiedKFold(yy, n_folds=n_folds, shuffle=True)
            clf = grid_search.GridSearchCV(svmclf, parameters, scoring='f1', n_jobs=-1, cv=cv)
            with warnings.catch_warnings(): 
                warnings.simplefilter("ignore")
                clf.fit(XX, yy)
            print clf.best_params_, clf.best_score_
            self.clfs.append(clf.best_estimator_)


    def predict_prob(self, feature):
        if not feature.any():
            return [0]*len(self.clfs)
        return [clf.decision_function(feature)[0] for clf in self.clfs]
        
