''' Quantile regression trees '''

from .._model import BaseModel
from .tree_builder import TreeBuilder

from typing import Optional
import numpy as np

class QuantileRegressionTree(BaseModel):
    ''' A decision tree for regression with prediction intervals.

    Examples:
        Fitting and predicting follows scikit-learn syntax:
        >>> from doubt.datasets import Concrete
        >>> X, y = Concrete().split()
        >>> tree = QuantileRegressionTree()
        >>> tree.fit(X, y).predict(X).shape
        (1030,)
        >>> tree.predict(np.ones(8))
        28.58912234

        Instead of only returning the prediction, we can also return a
        prediction interval:
        >>> tree.predict(np.ones(8), uncertainty = 0.05)
        (28.58912234, array([14.31610729, 40.65753788]))
    '''
    def __init__(self, min_samples_leaf: int = 5):
        self._tree = None

    def fit(self, X, y):
        self._tree = TreeBuilder().build(X, y)
        return self

    def predict(self, X, uncertainty: Optional[float] = None):
        onedim = (len(X.shape) == 1)
        if onedim: X = np.expand_dims(X, 0)

        values = [self._tree.find_leaf(x).vals for x in X]

        preds = np.array([val.mean() for val in values])
        if onedim: preds = preds[0]
        if uncertainty is not None:
            lower = uncertainty / 2
            upper = 1 - lower
            intervals = np.array([np.quantile(val, q = [lower, upper]) 
                                  for val in values])
            if onedim: intervals = intervals[0]
            return preds, intervals
        else:
            return preds
