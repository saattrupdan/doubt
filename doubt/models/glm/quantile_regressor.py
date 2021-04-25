'''Quantile regression for generalised linear models'''

from .._model import BaseModel
from .quantile_loss import smooth_quantile_loss
from .quantile_loss import quantile_loss

from sklearn.linear_model._glm import GeneralizedLinearRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Sequence, Tuple, Union, Optional
from scipy.optimize import minimize


FloatMatrix = Sequence[Sequence[float]]
FloatArray = Sequence[float]


class QuantileRegressor(BaseModel):
    '''Quantile regression for generalised linear models.

    This uses BFGS optimisation of the smooth quantile loss from [1].

    Args:
        max_iter (int):
            The maximal number of iterations to train the model for. Defaults
            to 10,000.
        uncertainty (float):
            The uncertainty in the prediction intervals. Must be between 0
            and 1. Defaults to 0.05.
        quantiles (sequence of floats or None, optional):
            List of quantiles to output, as an alternative to the
            `uncertainty` argument, and will not be used if that argument
            is set. If None then `uncertainty` is used. Defaults to None.
        alpha (float, optional):
            Smoothing parameter. Defaults to 0.4.


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
            >>> pred, interval
            (78.50224243713615, array([ 19.27889046, 172.71407256]))

    Sources:
        [1]: Songfeng Zheng (2011). Gradient Descent Algorithms for
             Quantile Regression With Smooth Approximation. International
             Journal of Machine Learning and Cybernetics.
    '''
    def __init__(self,
                 model: Union[LinearRegression, GeneralizedLinearRegressor],
                 max_iter: Optional[int] = None,
                 uncertainty: float = 0.05,
                 quantiles: Optional[Sequence[float]] = None,
                 alpha: float = 0.4):

        self.uncertainty = uncertainty
        self.alpha = alpha

        self._model = model
        self._scaler = StandardScaler()

        # Set `max_iter` to be the model's `max_iter` attribute if it exists,
        # and otherwise default to 10,000
        if max_iter is None:
            if hasattr(model, 'max_iter'):
                self.max_iter = model.max_iter
            else:
                self.max_iter = 10_000
        else:
            self.max_iter = max_iter

        # Default quantiles to the prediction interval
        if quantiles is None:
            self.quantiles = [self.uncertainty / 2, 1 - (self.uncertainty / 2)]
        else:
            self.quantiles = list(quantiles)

        # Initialise empty inverse link function and weights
        self._inverse_link_function = None
        self._weights = {q: None for q in self.quantiles}

    def _objective_function(self,
                            beta: np.ndarray,
                            X: np.ndarray,
                            y: np.ndarray,
                            quantile: float,
                            inverse_link_function: callable) -> float:
        '''Function used to optimise the quantile loss'''
        predictions = inverse_link_function(X @ beta)
        loss = smooth_quantile_loss(predictions=predictions,
                                    targets=y,
                                    quantile=quantile,
                                    alpha=self.alpha)
        return loss

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
        # Convert inputs to Numpy arrays
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # Fit the scaler and scale `X`
        self._scaler.fit(X_arr)
        X_arr = self._scaler.transform(X_arr)

        # Fit the underlying model
        self._model.fit(X_arr, y_arr)

        # Add constant feature
        X_arr = np.concatenate((X_arr, np.ones((X_arr.shape[0], 1))), axis=1)

        # When a Scikit-Learn GLM is fitted then its link function is set to
        # the attribute _link_instance. As the LinearRegression class does not
        # subclass the GeneralizedLinearRegressor class then we deal with this
        # case separately
        try:
            link = self._model._link_instance
            self._inverse_link_function = link.inverse
        except AttributeError:
            self._inverse_link_function = lambda x: x

        # Use the underlying GLM's parameters for the initial guess of the
        # quantile weights
        model_weights = self._model.coef_
        model_intercept = self._model.intercept_
        beta_init = np.concatenate((model_weights, [model_intercept]))

        # Fit all quantile estimates
        for q in self.quantiles:

            # Initialise random seed, which is used by SciPy
            if random_seed is not None:
                np.random.seed(random_seed)

            args = (X_arr, y_arr, q, self._inverse_link_function)
            result = minimize(self._objective_function,
                              beta_init,
                              args=args,
                              method='BFGS',
                              options={'maxiter': self.max_iter})
            self._weights[q] = result.x

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

        # Standardise the feature matrix
        X_arr = self._scaler.transform(X_arr)

        # Compute the predictions from the underlying model
        preds = self._model.predict(X_arr)

        # Add constant feature
        X_arr = np.concatenate((X_arr, np.ones((X_arr.shape[0], 1))), axis=1)

        # Get the prediction for the lower- and upper quantiles
        quantile_vals = [self._inverse_link_function(X_arr @ self._weights[q])
                         for q in self._weights.keys()]

        # Concatenate the quantiles to get the intervals
        quantile_vals = np.stack(quantile_vals, axis=1).squeeze()

        # If we started with a one-dimensional input, ensure that the output is
        # also one-dimensional
        if onedim:
            preds = preds.item()

        return preds, quantile_vals

    def score(self, X: Sequence[float], y: Sequence[float]) -> float:
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
        # Convert inputs to Numpy arrays
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # If `uncertainty` has not been specified then simply use the `score`
        # method from the underlying LinearRegression model, which returns the
        # R^2 value
        if self.uncertainty is None:
            return self._linreg.score(X_arr, y_arr)

        # If `uncertainty` has been specified then compute pinball loss
        else:
            # Get the predictions
            _, quantile_vals = self.predict(X_arr)
            losses = [quantile_loss(y_arr, quantile_vals[:, i], q)
                      for i, q in enumerate(self.quantiles)]
            return -np.mean(losses)
