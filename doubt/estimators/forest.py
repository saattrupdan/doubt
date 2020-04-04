''' Quantile random forests '''

from ._estimator import BaseEstimator

from ._forest import _branch
from ._forest import _predict

from typing import Union
from typing import Optional

import numpy as np
import pandas as pd
import scipy.optimize as opt

from anytree import Node
from anytree.exporter import DotExporter

from functools import partial
from collections import defaultdict

from joblib import Parallel
from joblib import delayed

from tqdm.auto import tqdm

class DecisionTree(BaseEstimator):
    ''' A decision tree for regression.

    Args:
        method (str): 
            The method used. Can be any of the following, defaults to 'cart':
                'cart': Classification and Regression Trees, see [2]
                'prim': Patient Rule Induction Method, see [3]
                'mars': Multivariate Adaptive Regression Splines, see [4]
                'hme': Hierarchical Mixtures of Experts, see [5]
        min_samples_leaf (int):
            The minimum number of unique target values in a leaf. Defaults
            to 5.

    Attributes:
        peeling_ratio (float): 
            The percentage of the training set that will be "peeled" at 
            every step when using the PRIM method. Only relevant when
            `method`='prim'. Defaults to 0.1.

    Methods:
        fit(X, y): 
        predict(X, y):
        plot():


    References:
        .. [1] Hastie, T., Tibshirani, R., & Friedman, J. (2009). 
               The Elements of Statistical Learning: Data Mining, Inference, 
               and Prediction. Springer Science & Business Media. 
        .. [2] Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. 
               (1984). Classification and regression trees. 
               Wadsworth & Brooks. Cole Statistics/Probability Series. 
        .. [3] Friedman, J. H., & Fisher, N. I. (1999). 
               Bump Hunting in High-Dimensional Data. 
               Statistics and Computing, 9(2), 123-143.
        .. [4] Friedman, J. H. (1991). 
               Multivariate Adaptive Regression Splines. 
               The Annals of Statistics, 1-67.
        .. [5] Jordan, M. I., & Jacobs, R. A. (1994). 
               Hierarchical Mixtures of Experts and the EM Algorithm. 
               Neural Computation, 6(2), 181-214.
    Examples:
        A simple example where we fit a decision tree and predict a value 
        from the same training set:

        >>> from doubt.datasets import TehranHousing 
        >>> X, y = TehranHousing().split()
        >>> y = y[:, 0]
        >>> tree = DecisionTree().fit(X, y)
        >>> tree.predict(X[0]).astype(np.int32)
        44952

        We can also predict multiple values at once:

        >>> tree.predict(X[0:3]).shape
        (3,)

        The tree can also be called directly, as an alias for `predict`:

        >>> tree(X[0:3]).shape
        (3,)
    '''
    def __init__(self, method: str = 'cart', min_samples_leaf: int = 5):
        self._root: Node
        self.method = method
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        ''' Fit a decision tree to the data. '''

        # Call the branching function recursively
        self._root = _branch(X, y, min_samples_leaf = self.min_samples_leaf)
        return self

    def export_graph(self, path: str):
        ''' Save an image of the decision tree to disk. '''

        if self._root is None:
            raise RuntimeError('No tree found. You might need to fit it '\
                               'to a data set first?')

        DotExporter(self._root).to_picture(path)
        return self

    def predict(self, X, quantile: Optional[float] = None):
        ''' Predict the response values of a given feature matrix. '''

        if self._root is None:
            raise RuntimeError('No tree found. You might need to fit it '\
                               'to a data set first?')

        # Ensure that the input is two-dimensional
        if len(X.shape) == 1: X = np.expand_dims(X, 0)

        if quantile is None: 
            quantile = -1.
        elif quantile < 0. or quantile > 1.:
            raise RuntimeError('Quantiles must be between 0 and 1 inclusive')

        return _predict(self._root, X, quantile = quantile)

class QuantileRandomForest(BaseEstimator):
    def __init__(self, 
        n_estimators: int = 10, 
        method: str = 'cart',
        min_samples_leaf: int = 5, 
        n_jobs: int = -1):

        self.n_estimators = n_estimators
        self.method = method
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs

        self._estimators = n_estimators * [
            DecisionTree(method = method, min_samples_leaf = min_samples_leaf)
        ]

    def fit(self, X, y):
        ''' Fit decision trees in parallel. '''
        nrows = X.shape[0]

        # Get bootstrap resamples of the data set
        bidxs = np.random.choice(nrows, size = (self.n_estimators, nrows), 
                                 replace = True)

        # Fit trees in parallel on the bootstrapped resamples
        pbar = tqdm(self._estimators, desc = 'Growing trees')
        with Parallel(n_jobs = self.n_jobs, backend = 'threading') as parallel:
            self._estimators = parallel(
                delayed(estimator.fit)(X[bidxs[b, :], :], y[bidxs[b, :]])
                for b, estimator in enumerate(pbar)
            )
        return self

    def predict(self, X, uncertainty: Optional[float] = None,
                quantile: Optional[float] = None):
        ''' Perform predictions. '''

        if uncertainty is not None and quantile is not None:
            raise RuntimeError('Both uncertainty and quantile can not be set')

        # Ensure that X is two-dimensional
        if len(X.shape) == 1: X = np.expand_dims(X, 0)

        with Parallel(n_jobs = self.n_jobs, backend = 'threading') as parallel:

            predictions = parallel(delayed(estimator.predict)(X, quantile)
                                   for estimator in self._estimators)

            if uncertainty is not None:
                lower_quantile = (1 - uncertainty) / 2
                lower = parallel(delayed(estimator.predict)(X, lower_quantile)
                                 for estimator in self._estimators)
                upper_quantile = (1 + uncertainty) / 2
                upper = parallel(delayed(estimator.predict)(X, upper_quantile)
                                 for estimator in self._estimators)

                return (np.mean(predictions, axis = 0), 
                        np.mean(lower, axis = 0),
                        np.mean(upper, axis = 0))

        return np.mean(predictions, axis = 0)
