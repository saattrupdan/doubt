'''Linear models'''

from .._model import BaseModel

from sklearn.linear_model import LinearRegression
from statsmodels.regression.quantile_regression import QuantReg
import numpy as np

from typing import Sequence, Union, Tuple


FloatArray = Union[Sequence[float], np.ndarray]


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
    def __init__(self,
                 uncertainty: float = 0.05,
                 max_iter: int = 10000,
                 n_jobs: int = -1):
        self.uncertainty = uncertainty
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self._linreg = LinearRegression(n_jobs=n_jobs)
        self._q_bias = np.empty((2,))
        self._q_slope = None

    def __repr__(self) -> str:
        return (f'QuantileLinearRegression(uncertainty={self.uncertainty}, '
                f'max_iter={self.max_iter}, n_jobs={self.n_jobs})')

    def fit(self, X: FloatArray, y: FloatArray):
        '''Fit the model.

        Args:
            X (float array):
                The array containing the data set, either of shape (n,) or
                (n, f), with n being the number of samples and f being the
                number of features.
            y (float array):
                The target array, of shape (n,).
        '''
        # Convert inputs to NumPy arrays
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # Fit the linear regression to the inputs
        self._linreg.fit(X_arr, y_arr)

        # Attach a bias feature to feature matrix
        n = X_arr.shape[0]
        X_arr = np.concatenate((np.ones((n, 1)), X_arr), axis=1)

        # Fit quantile regression model from `statsmodels` if `uncertainty` is
        # not None
        if self.uncertainty is not None:

            # Get the two quantiles of interest
            lower_q = self.uncertainty / 2.
            upper_q = 1. - lower_q

            # Initialise the quantile regression
            statsmodels_qreg = QuantReg(y_arr, X_arr)

            # Fit the two quantiles and save the results in `self._q_bias` and
            # `self._q_slope`
            self._q_slope = np.empty((2, X_arr.shape[1] - 1))
            for i, quantile in enumerate([lower_q, upper_q]):
                try:
                    result = statsmodels_qreg.fit(q=quantile,
                                                  max_iter=self.max_iter)

                # Statsmodels cannot model quantile regression on singular
                # feature matrices. That library raises a non-informative error
                # message, so raise instead a more descriptive one in that case
                except ValueError as e:
                    if str(e).startswith('operands could not be broadcast '
                                         'together with shapes'):
                        raise RuntimeError('Your feature matrix seems to be '
                                           'singular, in which case quantile '
                                           'linear regression cannot be '
                                           'performed. A common reason for '
                                           'this is duplicated or '
                                           'near-duplicated features.')
                    else:
                        raise e

                self._q_bias[i] = result.params[0]
                self._q_slope[i] = result.params[1:]

        return self

    def predict(self,
                X: FloatArray) -> Tuple[Union[float, np.ndarray], np.ndarray]:
        '''Compute model predictions.

        Args:
            X (float array):
                The array containing the data set, either of shape (n,) or
                (n, f), with n being the number of samples and f being the
                number of features.

        Returns:
            pair of float arrays:
                The predictions, of shape (n,), and the prediction intervals,
                of shape (n, 2).
        '''
        # Convert inputs to NumPy array
        X_arr = np.asarray(X)

        # If input is one-dimensional, then add a dimension to it
        onedim = (len(X_arr.shape) == 1)
        if onedim:
            X_arr = np.expand_dims(X_arr, 0)

        # Compute the linear regression predictions
        preds = self._linreg.predict(X_arr)

        # Get the prediction for the lower- and upper quantiles
        lower = np.sum(self._q_slope[0] * X_arr, axis=1) + self._q_bias[0]
        upper = np.sum(self._q_slope[1] * X_arr, axis=1) + self._q_bias[1]

        # Concatenate the quantiles to get the intervals
        intervals = np.stack([lower, upper], axis=1).squeeze()

        # If we started with a one-dimensional input, ensure that the output is
        # also one-dimensional
        if onedim:
            preds = preds.item()

        return preds, intervals

    @staticmethod
    def _pinball_loss(target: FloatArray,
                      prediction: FloatArray,
                      quantile: float) -> float:
        '''Implementation of the pinball loss.

        This is computed as quantile(target - prediction) if
        target >= prediction, and (1 - quantile)(prediction - target)
        otherwise.

        Args:
            target (float array):
                The target array, of shape (n,).
            prediction (float array):
                The predictions, of shape (n,).
            quantile (float):
                The quantile. Needs to be between 0.0 and 1.0.

        Returns:
            float: The negative pinball loss.
        '''
        # Convert inputs to NumPy arrays
        target_arr = np.asarray(target)
        prediction_arr = np.asarray(prediction)

        # Compute the residuals
        res = target_arr - prediction_arr

        # Compute the pinball loss
        loss = np.mean(np.maximum(res, np.zeros_like(res)) * quantile +
                       np.maximum(-res, np.zeros_like(res)) * (1 - quantile))
        return float(loss)

    def score(self, X: FloatArray, y: FloatArray) -> float:
        '''Compute either the R^2 value or the negative pinball loss.

        If `uncertainty` is not set in the constructor then the R^2 value will
        be returned, and otherwise the mean of the two negative pinball losses
        corresponding to the two quantiles will be returned.

        The pinball loss is computed as quantile * (target - prediction) if
        target >= prediction, and (1 - quantile)(prediction - target)
        otherwise.

        Args:
            X (float array):
                The array containing the data set, either of shape (n,) or
                (n, f), with n being the number of samples and f being the
                number of features.
            y (float array):
                The target array, of shape (n,).

        Returns:
            float: The negative pinball loss.
        '''
        # If `uncertainty` has not been specified then simply use the `score`
        # method from the underlying LinearRegression model, which returns the
        # R^2 value
        if self.uncertainty is None:
            return self._linreg.score(X, y)

        # If `uncertainty` has been specified then compute pinball loss
        else:
            # Convert inputs to NumPy arrays
            X_arr = np.asarray(X)
            y_arr = np.asarray(y)

            # Define the lower and upper quantiles
            lower_q = self.uncertainty / 2.
            upper_q = 1. - lower_q

            # Get the predictions
            _, intervals = self.predict(X_arr)
            lower_loss = self._pinball_loss(y_arr, intervals[:, 0], lower_q)
            upper_loss = self._pinball_loss(y_arr, intervals[:, 1], upper_q)
            return -(lower_loss + upper_loss) / 2
