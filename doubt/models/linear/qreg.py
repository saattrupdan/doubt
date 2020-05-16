''' Linear models '''

from .._model import BaseModel

from sklearn.linear_model import LinearRegression
from statsmodels.regression.quantile_regression import QuantReg
import numpy as np

from typing import Optional
from typing import Sequence

FloatArray = Sequence[float]

class QuantileLinearRegression(BaseModel):
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
        onedim = (len(X.shape) == 1)
        if onedim: X = np.expand_dims(X, 0)
        preds = self.linreg.predict(X)
        lower = np.sum(self.q_slope[0] * X, axis = 1) + self.q_bias[0]
        upper = np.sum(self.q_slope[1] * X, axis = 1) + self.q_bias[1]
        intervals = np.stack([lower, upper], axis = 1)
        return preds.squeeze(), intervals.squeeze()

if __name__ == '__main__':
    from doubt.datasets import Concrete
    X, y = Concrete().split()
    qreg = QuantileLinearRegression(uncertainty = 0.05)
    qreg.fit(X, y)
    print(qreg.predict(X))
