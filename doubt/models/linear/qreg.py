''' Linear models '''

# TODO: Add documentation

from .._model import BaseModel

from sklearn.linear_model import LinearRegression
from statsmodels.regression.quantile_regression import QuantReg
import numpy as np

from typing import Optional
from typing import Sequence

FloatArray = Sequence[float]

class QuantileLinearRegression(BaseModel):
    ''' Quantile linear regression model.

    Examples:
        Fitting and predicting follows scikit-learn syntax:
        >>> from doubt.datasets import Concrete
        >>> X, y = Concrete().split(random_seed = 42)
        >>> model = QuantileLinearRegression(uncertainty = 0.05)
        >>> model.fit(X, y).predict(X)[0].shape
        (1030,)
        >>> model.predict([500, 0, 0, 100, 2, 1000, 500, 20])
        (52.672378992388026, array([ 30.41893262, 106.94239059]))
    '''
    def __init__(self, uncertainty: 0.05, max_iter: int = 10000,
                 n_jobs: int = -1):
        self.uncertainty = uncertainty
        self.max_iter = max_iter
        self.linreg = LinearRegression(n_jobs = n_jobs)
        self.q_bias = np.empty((2,))
        self.q_slope: Optional[FloatArray] = None

    def fit(self, X, y):
        self.linreg.fit(X, y)

        n = X.shape[0]
        X = np.concatenate((np.ones((n, 1)), X), axis = 1)

        if self.uncertainty is not None:
            statsmodels_qreg = QuantReg(y, X)
            self.q_slope = np.empty((2, X.shape[1] - 1))
            lower_q = self.uncertainty / 2.
            upper_q = 1. - lower_q
            for i, quantile in enumerate([lower_q, upper_q]):
                result = statsmodels_qreg.fit(q = quantile, 
                                              max_iter = self.max_iter)
                self.q_bias[i] = result.params[0]
                self.q_slope[i] = result.params[1:]

        return self

    def predict(self, X):
        X = np.asarray(X)
        onedim = (len(X.shape) == 1)
        if onedim: X = np.expand_dims(X, 0)
        preds = self.linreg.predict(X).squeeze()
        lower = np.sum(self.q_slope[0] * X, axis = 1) + self.q_bias[0]
        upper = np.sum(self.q_slope[1] * X, axis = 1) + self.q_bias[1]
        intervals = np.stack([lower, upper], axis = 1).squeeze()
        if onedim: preds = preds.item()
        return preds, intervals

if __name__ == '__main__':
    from doubt.datasets import Concrete
    X, y = Concrete().split()
    qreg = QuantileLinearRegression(uncertainty = 0.05)
    qreg.fit(X, y)
    print(qreg.predict(X))
