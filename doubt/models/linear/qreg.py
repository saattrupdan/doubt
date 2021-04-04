'''Linear models'''

from .._model import BaseModel

from sklearn.linear_model import LinearRegression
from statsmodels.regression.quantile_regression import QuantReg
import numpy as np

from typing import Optional, Sequence

FloatArray = Sequence[float]


class QuantileLinearRegression(BaseModel):
    '''Quantile linear regression model.

    Args:
        uncertainty (float):
            The uncertainty in the prediction intervals. Must be between 0
            and 1. Defaults to 0.05.
        max_iter (int):
            The maximal number of iterations to train the model for. Defaults
            to 10,000.
        n_jobs (int):
            The number of CPU cores to run in parallel when training. If set
            to -1 then all CPU cores will be used. Defaults to -1.

    Methods:
        fit(X, y) -> self
        predict(X) -> tuple

    Examples:
        Fitting and predicting follows scikit-learn syntax:
        >>> from doubt.datasets import Concrete
        >>> X, y = Concrete().split(random_seed=42)
        >>> model = QuantileLinearRegression(uncertainty=0.05)
        >>> model.fit(X, y).predict(X)[0].shape
        (1030,)
        >>> pred, interval = model.predict([500, 0, 0, 100, 2, 1000, 500, 20])
        >>> pred, interval[0], interval[1]
        (52.672378992388026, 30.418533804253457, 106.94238881241851)
    '''
    def __init__(self, uncertainty: float = 0.05, max_iter: int = 10000,
                 n_jobs: int = -1):
        self.uncertainty = uncertainty
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self._linreg = LinearRegression(n_jobs=n_jobs)
        self._q_bias = np.empty((2,))
        self._q_slope: Optional[FloatArray] = None

    def __repr__(self) -> str:
        return (f'QuantileLinearRegression(uncertainty={self.uncertainty}, '
                f'max_iter={self.max_iter}, n_jobs={self.n_jobs})')

    def fit(self, X: FloatArray, y: FloatArray):
        '''Fit the model.

        Args:
            X (float array):
                The feature matrix.
            y (float array):
                The target matrix.
        '''
        self._linreg.fit(X, y)

        n = X.shape[0]
        X = np.concatenate((np.ones((n, 1)), X), axis=1)

        if self.uncertainty is not None:
            statsmodels_qreg = QuantReg(y, X)
            self._q_slope = np.empty((2, X.shape[1] - 1))
            lower_q = self.uncertainty / 2.
            upper_q = 1. - lower_q
            for i, quantile in enumerate([lower_q, upper_q]):
                result = statsmodels_qreg.fit(q=quantile,
                                              max_iter=self.max_iter)
                self._q_bias[i] = result.params[0]
                self._q_slope[i] = result.params[1:]

        return self

    def predict(self, X: FloatArray):
        '''Compute model predictions.

        Args:
            X (float array):
                The array containing the data set, either of shape (n,) or
                (n, f), with n being the number of samples and f being the
                number of features.

        Returns:
            pair of float arrays:
                The predictions and the prediction intervals
        '''
        X = np.asarray(X)
        onedim = (len(X.shape) == 1)
        if onedim:
            X = np.expand_dims(X, 0)
        preds = self._linreg.predict(X).squeeze()
        lower = np.sum(self._q_slope[0] * X, axis=1) + self._q_bias[0]
        upper = np.sum(self._q_slope[1] * X, axis=1) + self._q_bias[1]
        intervals = np.stack([lower, upper], axis=1).squeeze()
        if onedim:
            preds = preds.item()
        return preds, intervals
