''' Quantile regression forests '''

from .._model import BaseModel
from .tree import QuantileRegressionTree

from typing import Optional
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed

class QuantileRegressionForest(BaseModel):
    ''' A random forest for regression which can output quantiles as well.

    Examples:
        Fitting and predicting follows scikit-learn syntax:
        >>> from doubt.datasets import Concrete
        >>> X, y = Concrete().split()
        >>> forest = QuantileRegressionForest(random_seed = 42)
        >>> forest.fit(X, y).predict(X).shape
        (1030,)
        >>> forest.predict(np.ones(8))
        8.08755348

        Instead of only returning the prediction, we can also return a
        prediction interval:
        >>> forest.predict(np.ones(8), uncertainty = 0.05)
        (8.08755348, array([ 6.26733684, 12.63809508]))
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
        n = X.shape[0]

        if self.random_seed is not None: np.random.seed(self.random_seed)

        # Get bootstrap resamples of the data set
        bidxs = np.random.choice(n, size = (self.n_estimators, n), 
                                 replace = True)

        # Fit trees in parallel on the bootstrapped resamples
        pbar = tqdm(self._estimators, desc = 'Growing trees')
        with Parallel(n_jobs = self.n_jobs, backend = 'threading') as parallel:
            self._estimators = parallel(
                delayed(estimator.fit)(X[bidxs[b, :], :], y[bidxs[b, :]])
                for b, estimator in enumerate(pbar)
            )
        return self

    def predict(self, X, uncertainty: Optional[float] = None):
        ''' Perform predictions. '''

        # Ensure that X is two-dimensional
        onedim = (len(X.shape) == 1)
        if onedim: X = np.expand_dims(X, 0)

        with Parallel(n_jobs = self.n_jobs, backend = 'threading') as parallel:

            preds = parallel(
                delayed(estimator.predict)(X, uncertainty)
                for estimator in self._estimators
            )
            if uncertainty is not None:
                intervals = np.concatenate([interval for _, interval in preds])
                intervals = np.mean(intervals, axis = 0)
                preds = np.concatenate([pred for pred, _ in preds])
                preds = np.mean(preds, axis = 0)
                return preds, intervals
            
            else:
                preds = np.mean(preds, axis = 0)
                if onedim: preds = preds[0]
                return preds
