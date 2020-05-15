''' Quantile regression forests '''

from .._model import BaseModel
from .tree import QuantileRegressionTree

from typing import Optional
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed

class QuantileRegressionForest(BaseModel):
    ''' A random forest for regression which can output quantiles as well.

    >>> from doubt.datasets import Concrete
    >>> X, y = Concrete().split()
    >>> forest = QuantileRegressionForest(random_seed = 42)
    >>> forest.fit(X, y).predict(X).shape
    (1030,)
    >>> forest.predict(np.ones(8)).round()
    array([8.])
    '''
    def __init__(self, 
        n_estimators: int = 10, 
        min_samples_leaf: int = 5, 
        n_jobs: int = -1,
        random_seed: Optional[int] = None):

        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_seed = random_seed

        self._estimators = n_estimators * [
            QuantileRegressionTree(min_samples_leaf = min_samples_leaf)
        ]

    def fit(self, X, y):
        ''' Fit decision trees in parallel. '''
        nrows = X.shape[0]

        if self.random_seed is not None: np.random.seed(self.random_seed)

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
                        (np.mean(lower, axis = 0),
                         np.mean(upper, axis = 0)))

        return np.mean(predictions, axis = 0)
