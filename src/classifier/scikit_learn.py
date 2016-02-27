from __future__ import absolute_import
import copy
import inspect
import types
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
# from ..utils.np_utils import to_categorical
# from ..models import Sequential


class BaseWrapper(object):

    """
    Base class for the Keras scikit-learn wrapper.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    build_fn: callable function or class instance,  optional
        Implementing the logic of the model.
    sk_params: model parameters & fitting parameters
    ----------
    The build_fn should construct, compile and return a Keras model, which
    will then be used to fit data or predict unknow data. One of the following
    three values could be passed to build_fn:
    1. A function instance
    2. An instance of a class that implements the __call__ function
    3. None. This means you implement a class that inherits either
    KerasClassifier or KerasRegressor. The __call__ function of the class will
    then be treated as the default build_fn

    'sk_params' takes both model parameters and fitting parameters. Legal model
    parameters are the arguments of 'build_fn'. Note that like all other
    estimators in scikit_learn, 'build_fn' should provide defalult velues for
    its arguments, so that you could create the estimator without passing any
    values to 'sk_params'.

    'sk_params' could also accept parameters for calling 'fit', 'predict',
    'predict_proba', and 'score' function (e.g., nb_epoch, batch_size).
    fitting (predicting) parameters are adopts in the following order:

    1. Values passed to the dictionary arguments of
    'fit', 'predict', 'predict_proba', and 'score' functions
    2. Values passed to 'sk_params'
    3. The default values of the keras.models.Sequential's
    'fit', 'predict', 'predict_proba' and 'score' functions

    When using scikit_learn's grid_search api, legal tunable parameters are
    those you could pass to 'sk_params', including fitting parameters.
    In other words, you could use grid_search to search for the best
    batch_size or nb_epoch as well as the model parameters.
    """

    def __init__(self, build_fn=None, **sk_params):
        self.build_fn = build_fn
        self.sk_params = sk_params

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Dictionary of parameter names mapped to their values.
        """
        res = copy.deepcopy(self.sk_params)
        res.update({'build_fn': self.build_fn})
        return res

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        params: dict
            Dictionary of parameter names mapped to their values.

        Returns
        -------
        self
        """
        self.sk_params.update(params)
        return self

    def fit(self, X, y, **kwargs):
        """
        Construct a new model with build_fn and
        fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training samples where n_samples in the number of samples
            and n_features is the number of features.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        kwargs: dictionary arguments
            Legal arguments are the arguments of Sequential.fit

        Returns
        -------
        history : object
            Returns details about the training history at each epoch.
        """

        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType):
            self.model = self.build_fn(
                **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        if self.model.loss.__name__ == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)

        history = self.model.fit(X, y, **fit_args)

        return history

    def filter_sk_params(self, fn, override={}):
        """
        Filter sk_params and return those in fn's arguments

        Parameters
        ----------
        fn : arbitrary function
        override: dictionary,
            values to overrid sk_params

        Returns
        -------
        res : dictionary
            dictionary containing variabls in both sk_params
            and fn's arguments.
        """
        res = {}
        fn_args = inspect.getargspec(fn)[0]
        for name, value in self.sk_params.items():
            if name in fn_args:
                res.update({name: value})
        res.update(override)
        return res


class KerasClassifier(BaseWrapper):

    """
    Implementation of the scikit-learn classifier API for Keras.
    """

    def predict(self, X, **kwargs):
        """
        Returns the class predictions for the given test data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples where n_samples in the number of samples
            and n_features is the number of features.
        kwargs: dictionary arguments
            Legal arguments are the arguments of Sequential.predict_classes
        Returns
        -------
        preds : array-like, shape = (n_samples)
            Class predictions.
        """
        kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
        return self.model.predict_classes(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        """
        Returns class probability estimates for the given test data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples where n_samples in the number of samples
            and n_features is the number of features.
        kwargs: dictionary arguments
            Legal arguments are the arguments of Sequential.predict_classes

        Returns
        -------
        proba : array-like, shape = (n_samples, n_outputs)
            Class probability estimates.
        """
        kwargs = self.filter_sk_params(Sequential.predict_proba, kwargs)
        return self.model.predict_proba(X, **kwargs)

    def score(self, X, y, **kwargs):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples where n_samples in the number of samples
            and n_features is the number of features.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        kwargs: dictionary arguments
            Legal arguments are the arguments of Sequential.evaluate

        Returns
        -------
        score : float
            Mean accuracy of predictions on X wrt. y.
        """
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)
        kwargs.update({'show_accuracy': True})
        loss, accuracy = self.model.evaluate(X, y, **kwargs)
        return accuracy


class KerasRegressor(BaseWrapper):

    """
    Implementation of the scikit-learn regressor API for Keras.
    """

    def predict(self, X, **kwargs):
        """
        Returns predictions for the given test data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples where n_samples in the number of samples
            and n_features is the number of features.
        kwargs: dictionary arguments
            Legal arguments are the arguments of Sequential.predict
        Returns
        -------
        preds : array-like, shape = (n_samples)
            Predictions.
        """
        kwargs = self.filter_sk_params(Sequential.predict, kwargs)
        return self.model.predict(X, **kwargs)

    def score(self, X, y, **kwargs):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples where n_samples in the number of samples
            and n_features is the number of features.
        y : array-like, shape = (n_samples)
            True labels for X.
        kwargs: dictionary arguments
            Legal arguments are the arguments of Sequential.evaluate

        Returns
        -------
        score : float
            Mean accuracy of predictions on X wrt. y.
        """
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)
        kwargs.update({'show_accuracy': False})
        loss = self.model.evaluate(X, y, **kwargs)
        return loss
