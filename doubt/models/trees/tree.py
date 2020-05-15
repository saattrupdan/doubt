''' Quantile regression trees '''

from .._model import BaseModel
from .tree_builder import TreeBuilder

from typing import Optional
import numpy as np

class QuantileRegressionTree(BaseModel):
    ''' A decision tree for regression which can output quantiles as well.

    >>> from doubt.datasets import Concrete
    >>> X, y = Concrete().split()
    >>> tree = QuantileRegressionTree()
    >>> tree.fit(X, y).predict(X).shape
    (1030,)
    >>> tree.predict(np.ones(8)).round()
    array([29.])
    '''
    def __init__(self, min_samples_leaf: int = 5):
        self._tree = None

    def fit(self, X, y):
        self._tree = TreeBuilder().build(X, y)
        return self

    def predict(self, X, quantile: Optional[float] = None):
        if len(X.shape) == 1: X = np.expand_dims(X, 0)
        values = [self._tree.find_leaf(x).vals for x in X]
        if quantile is None:
            return np.array([val.mean() for val in values])
        else:
            return np.array([np.quantile(val, q = quantile) for val in values])
