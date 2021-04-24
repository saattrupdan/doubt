'''Quantile regression for generalised linear models'''

from .._model import BaseModel
from .quantile_loss import quantile_loss

from sklearn.linear_model._glm import GeneralizedLinearRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Sequence, Tuple, Union, Optional
from scipy.optimize import minimize


FloatMatrix = Sequence[Sequence[float]]
FloatArray = Sequence[float]


class QuantileRegressor(BaseModel):
    '''Quantile regression for generalised linear models.

    Args:
        uncertainty (float):
            The uncertainty in the prediction intervals. Must be between 0
            and 1. Defaults to 0.05.
        max_iter (int):
            The maximal number of iterations to train the model for. Defaults
            to 10,000.

    Examples:
        Fitting and predicting follows scikit-learn syntax::

            >>> from doubt.datasets import Concrete
            >>> from sklearn.linear_model import PoissonRegressor
            >>> X, y = Concrete().split(random_seed=42)
            >>> model = QuantileRegressor(PoissonRegressor(max_iter=10_000),
            ...                           uncertainty=0.05)
            >>> model.fit(X, y).predict(X)[0].shape
            (1030,)
            >>> x = [500, 0, 0, 100, 2, 1000, 500, 20]
            >>> pred, interval = model.predict(x)
            >>> pred, interval[0], interval[1]
            (52.672378992388026, 30.418533804253457, 106.94238881241851)
    '''
    def __init__(self,
                 model: Union[LinearRegression, GeneralizedLinearRegressor],
                 uncertainty: float = 0.05,
                 quantiles: Optional[Sequence[float]] = None):

        self._model = model
        self.max_iter = model.max_iter
        self.uncertainty = uncertainty
        self._inverse_link_function = None

        if quantiles is None:
            self.quantiles = [self.uncertainty / 2, 1 - (self.uncertainty / 2)]
        else:
            self.quantiles = list(quantiles)

        self.weights = {q: None for q in self.quantiles}

    @staticmethod
    def _objective_function(beta: FloatArray,
                            X: FloatMatrix,
                            y: FloatArray,
                            quantile: float,
                            inverse_link_function: callable) -> float:
        '''Function used to optimise the quantile loss'''

        # Ensure that inputs are Numpy arrays
        beta_arr = np.asarray(beta)
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        predictions = inverse_link_function(X_arr @ beta_arr)
        return quantile_loss(predictions=predictions,
                             targets=y_arr,
                             quantile=quantile)

    def fit(self,
            X: FloatMatrix,
            y: FloatArray,
            random_seed: Optional[int] = None):
        '''Fit the model.

        Args:
            X (float matrix):
                The array containing the data set, either of shape (n,) or
                (n, f), with n being the number of samples and f being the
                number of features.
            y (float array):
                The target array, of shape (n,).
        '''
        # Initialise random number generator
        rng = np.random.default_rng(random_seed)

        # Convert inputs to Numpy arrays
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # Fit the underlying model
        self._model.fit(X_arr, y_arr)

        # When a Scikit-Learn GLM is fitted then its link function is set to
        # the attribute _link_instance. As the LinearRegression class does not
        # subclass the GeneralizedLinearRegressor class then we deal with this
        # case separately
        try:
            link = self._model._link_instance
            self._inverse_link_function = link.inverse
            self._inverse_link_gradient = link.inverse_derivative
        except AttributeError:
            self._inverse_link_function = lambda x: x
            self._inverse_link_gradient = lambda x: 1.

        # Fit all quantile estimates
        for q in self.quantiles:
            beta_init = rng.normal(0, 0.3, size=(X_arr.shape[1],))
            args = (X_arr, y_arr, q, self._inverse_link_function)
            result = minimize(self._objective_function, beta_init, args=args,
                              jac=True, options={'maxiter': self.max_iter})
            self.weights[q] = result.x

        return self

    def predict(self,
                X: FloatMatrix
                ) -> Tuple[Union[float, np.ndarray], np.ndarray]:
        '''Compute model predictions.

        Args:
            X (float matrix):
                The array containing the data set, either of shape (n,) or
                (n, f), with n being the number of samples and f being the
                number of features.

        Returns:
            pair of float arrays:
                The predictions, of shape (n,), and the prediction intervals,
                of shape (n, 2).
        '''
        # Convert inputs to Numpy array
        X_arr = np.asarray(X)

        # If input is one-dimensional, then add a dimension to it
        onedim = (len(X_arr.shape) == 1)
        if onedim:
            X_arr = np.expand_dims(X_arr, 0)

        # Get the prediction for the lower- and upper quantiles
        quantile_vals = [self._inverse_link_function(X_arr @ self.weights[q])
                         for q in self.weights.keys()]

        # Concatenate the quantiles to get the intervals
        quantile_vals = np.stack(quantile_vals, axis=1).squeeze()

        # Compute the predictions from the underlying model
        preds = self._model.predict(X_arr)

        # If we started with a one-dimensional input, ensure that the output is
        # also one-dimensional
        if onedim:
            preds = preds.item()

        return preds, quantile_vals
