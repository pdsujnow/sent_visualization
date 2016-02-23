import inspect
from keras.models import Sequential
from sklearn.base import BaseEstimator
import copy

class BaseWrapper(BaseEstimator):
    def __init__(self, build_fn, **model_params):
        self.build_fn = build_fn
        self.model_params = model_params 
        print "__init__"
        print self.model_params
        print

    def get_params(self, deep=True):
        res = copy.deepcopy(self.model_params)
        res.update({'build_fn':self.build_fn})
        print "get_params"
        print self.model_params
        print
        return res

    def set_params(self, **params):
        self.model_params.update(params) 
        print "set_params"
        print self.model_params
        print
        return self

class KerasClassifier(BaseWrapper):

    def get_fit_params(self, fn):
        fit_params = {}
        fit_params_name = inspect.getargspec(fn)
        for name,value in self.model_params.items():
            if name in fit_params_name:
                fit_params.update({name: value})
        return fit_params


    def fit(self, X, y):
        print "build"
        print self.model_params
        print
        self.model = self.build_fn(**self.model_params)
        return self.model.fit(X, y, **self.get_fit_params(Sequential.fit))

    def predict(self, X):
        return self.model.predict_class(X, **self.get_fit_params(Sequential.predict_class))

    def predict_proba(self, X):
        return self.model.predict_proba(X, **self.get_fit_params(Sequential.predict_proba))

