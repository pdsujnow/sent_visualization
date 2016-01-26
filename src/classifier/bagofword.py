import sys
sys.path.append('..')
from classifier_controler import Classifier
import numpy as np
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search
from word2vec.globvemongo import Globve


class BagofWord(Classifier):
    def init(self):
        self.globve = Globve()#wordvector model

    def un_init(self):
        del self.globve

    def text2vec(self, text):
        X = np.zeros(self.globve.dimension)
        for word in [w.decode('utf8') for w in text]:
            if word in self.globve:
                X = X+self.globve[word] 
        return X

    def fit(self, sentences, y, n_folds=5, random_state=0):

        X = np.array([self.text2vec(s) for s in sentences])

        np.random.seed(random_state)

        svmclf = svm.LinearSVC(dual=False)
        parameters = dict(C = np.logspace(-5,1, 8))
        clfs = []
        for i in range(y.shape[1]):
            yy = y[:, i]
            cv = StratifiedKFold(yy, n_folds=n_folds)
            clf = grid_search.GridSearchCV(svmclf, parameters, scoring='f1', n_jobs=-1, cv=cv)
            clf.fit(X, yy)
            clfs.append(clf.best_estimator_)

        return clfs

    def predict_prob(self, clf, feature):
        return clf.decision_function(feature)
