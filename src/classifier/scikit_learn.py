import inspect
from keras.models import Sequential
from sklearn.base import BaseEstimator
import copy
import types

class BaseWrapper(BaseEstimator):
    def __init__(self, build_fn=None, **sk_params):
        self.build_fn = build_fn
        self.sk_params = sk_params 
        #print "__init__"
        #print self.sk_params
        #print

    def get_params(self, deep=True):
        res = copy.deepcopy(self.sk_params)
        res.update({'build_fn':self.build_fn})
        #print "get_params"
        #print self.sk_params
        #print
        return res

    def set_params(self, **params):
        self.sk_params.update(params) 
        #print "set_params"
        #print self.sk_params
        #print
        return self

class KerasClassifier(BaseWrapper):

    def filter_sk_params(self, fn):
        res = {}
        fn_args = inspect.getargspec(fn)[0]
        for name,value in self.sk_params.items():
            if name in fn_args:
                res.update({name: value})
        return res

    def fit(self, X, y):
        #print "fit"
        #print self.sk_params
        #print

        # 1. self is a class that inherits KerasClassifier and implements __call__
        # 2. build_fn is a class that implements __call__
        # 3. build_fn is a function
        if self.build_fn is None: 
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif self.build_fn is not types.FunctionType:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))
        return self.model.fit(X, y, **self.filter_sk_params(Sequential.fit))

    def predict(self, X):
        return self.model.predict_classes(X, **self.filter_sk_params(Sequential.predict_classes))

    def predict_proba(self, X):
        return self.model.predict_proba(X, **self.filter_sk_params(Sequential.predict_proba))

