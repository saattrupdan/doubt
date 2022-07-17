"""Bootstrap wrapper for datasets and models"""

import copy
from types import MethodType
from typing import Callable, Optional, Tuple, Union

import numpy as np


class Boot:
    """Bootstrap wrapper for datasets and models.

    Datasets can be any sequence of numeric input, from which bootstrapped statistics
    can be calculated, with confidence intervals included.

    The models can be any model that is either callable or equipped with a `predict`
    method, such as all the models in `scikit-learn`, `pytorch` and `tensorflow`, and
    the bootstrapped model can then produce predictions with prediction intervals.

    The bootstrapped prediction intervals are computed using the an extension of method
    from [2] which also takes validation error into account. To remedy this, the .632+
    bootstrap estimate from [1] has been used. Read more in [3].

    Args:
        input (float array or model):
            Either a dataset to calculate bootstrapped statistics on, or an model for
            which bootstrapped predictions will be computed.
        random_seed (float or None):
            The random seed used for bootstrapping. If set to None then no seed will be
            set. Defaults to None.

    Examples:
        Compute the bootstrap distribution of the mean, with a 95% confidence
        interval::

            >>> from doubt.datasets import FishToxicity
            >>> X, y = FishToxicity().split()
            >>> boot = Boot(y, random_seed=42)
            >>> boot.compute_statistic(np.mean)
            (4.064430616740088, array([3.97621225, 4.16582087]))

        Alternatively, we can output the whole bootstrap distribution::

            >>> boot.compute_statistic(np.mean, n_boots=3, return_all=True)
            (4.064430616740088, array([4.05705947, 4.06197577, 4.05728414]))

        Wrap a scikit-learn model and get prediction intervals::

            >>> from sklearn.linear_model import LinearRegression
            >>> from doubt.datasets import PowerPlant
            >>> X, y = PowerPlant().split()
            >>> linreg = Boot(LinearRegression(), random_seed=42)
            >>> linreg = linreg.fit(X, y)
            >>> linreg.predict([10, 30, 1000, 50], uncertainty=0.05)
            (481.9968892065167, array([473.50425407, 490.14061895]))

    Sources:
        [1]: Friedman, J., Hastie, T., & Tibshirani, R. (2001). The elements of
             statistical learning (Vol. 1, No. 10). New York: Springer series in
             statistics.
        [2]: Kumar, S., & Srivistava, A. N. (2012). Bootstrap prediction intervals in
             non-parametric regression with applications to anomaly detection.
        [3]: https://saattrupdan.github.io/2020-03-01-bootstrap-prediction
    """

    def __init__(self, input: object, random_seed: Optional[float] = None):
        self.random_seed = random_seed

        # Input is a model
        if callable(input) or hasattr(input, "predict"):
            self._model = input
            self._model_fit_predict = MethodType(_model_fit_predict, self)
            self.fit = MethodType(fit, self)
            self.predict = MethodType(predict, self)
            type(self).__repr__ = MethodType(_model_repr, self)  # type: ignore

        # Input is a dataset
        elif hasattr(input, "__getitem__"):
            self.data = np.asarray(input)
            self.compute_statistic = MethodType(compute_statistic, self)
            type(self).__repr__ = MethodType(_dataset_repr, self)  # type: ignore

        else:
            raise TypeError("Input not recognised.")


def _model_fit_predict(
    self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """Fit the underlying model and perform predictions with it.

    This requires `self._model` to be set and that it is either callable or have a
    `predict` method.

    Args:
        X_train (float matrix):
            Feature matrix for training, of shape
            (n_train_samples, n_features).
        y_train (float array):
            Target array, of shape (n_train_samples,).
        X_test (float matrix):
            Feature matrix for predicting, of shape
            (n_test_samples, n_features).

    Returns:
        Numpy array:
            Predictions, of shape (n_test_samples,)
    """
    model = copy.deepcopy(self._model)
    model.fit(X_train, y_train)
    if callable(model):
        return model(X_test)
    else:
        return model.predict(X_test)


def _dataset_repr(self) -> str:
    return f"Boot(dataset_shape={self.data.shape}, " f"random_seed={self.random_seed})"


def _model_repr(self) -> str:
    model_name = self._model.__class__.__name__
    return f"Boot(model={model_name}, random_seed={self.random_seed})"


def compute_statistic(
    self,
    statistic: Callable[[np.ndarray], float],
    n_boots: Optional[int] = None,
    uncertainty: float = 0.05,
    quantiles: Optional[np.ndarray] = None,
    return_all: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """Compute bootstrapped statistic.

    Args:
        statistic (callable):
            The statistic to be computed on bootstrapped samples.
        n_boots (int or None, optional):
            The number of resamples to bootstrap. If None then it is set to the square
            root of the data set. Defaults to None
        uncertainty (float, optional):
            The uncertainty used to compute the confidence interval of the bootstrapped
            statistic. Not used if `return_all` is set to True or if `quantiles` is not
            None. Defaults to 0.05.
        quantiles (Numpy array or None, optional):
            List of quantiles to output, as an alternative to the `uncertainty`
            argument, and will not be used if that argument is set. If None then
            `uncertainty` is used. Defaults to None.
        return_all (bool, optional):
            Whether all bootstrapped statistics should be returned instead of the
            confidence interval. Defaults to False.

    Returns:
        a float or a pair of a float and an array of floats:
            The statistic, and if `uncertainty` is set then also the confidence
            interval, or if `quantiles` is set then also the specified quantiles, or if
            `return_all` is set then also all of the bootstrapped statistics.
    """
    # Initialise random number generator
    rng = np.random.default_rng(self.random_seed)

    # Compute the statistic
    stat = statistic(self.data)

    # Get the number of data points
    n = self.data.shape[0]

    # Set default value of the number of bootstrap samples if `n_boots` is not set
    if n_boots is None:
        n_boots = np.sqrt(n).astype(int)

    # Compute the bootstrapped statistics
    statistics = np.empty((n_boots,), dtype=float)
    for b in range(n_boots):
        boot_idxs = rng.choice(range(n), size=n, replace=True)
        statistics[b] = statistic(self.data[boot_idxs])

    if return_all:
        return stat, statistics
    else:
        # If uncertainty is set then set `quantiles` to be the two ends of the
        # confidence interval
        if uncertainty is not None:
            quantiles = [uncertainty / 2, 1.0 - (uncertainty / 2)]
        else:
            quantiles = list(quantiles)

        # Compute the quantile values
        quantile_vals = np.quantile(statistics, q=quantiles)
        return stat, quantile_vals


def predict(
    self,
    X: np.ndarray,
    n_boots: Optional[int] = None,
    uncertainty: Optional[float] = None,
    quantiles: Optional[np.ndarray] = None,
) -> Tuple[Union[float, np.ndarray], np.ndarray]:
    """Compute bootstrapped predictions.

    Args:
        X (float array):
            The array containing the data set, either of shape (f,) or (n, f), with n
            being the number of samples and f being the number of features.
        n_boots (int or None, optional):
            The number of resamples to bootstrap. If None then it is set to the square
            root of the data set. Defaults to None
        uncertainty (float or None, optional):
            The uncertainty used to compute the prediction interval of the bootstrapped
            prediction. If None then no prediction intervals are returned. Defaults to
            None.
        quantiles (sequence of floats or None, optional):
            List of quantiles to output, as an alternative to the `uncertainty`
            argument, and will not be used if that argument is set. If None then
            `uncertainty` is used. Defaults to None.

    Returns:
        float array or pair of float arrays:
            The bootstrapped predictions, and the confidence intervals if `uncertainty`
            is not None, or the specified quantiles if `quantiles` is not None.
    """
    # Initialise random number generator
    rng = np.random.default_rng(self.random_seed)

    # Ensure that input feature matrix is a Numpy array
    X = np.asarray(X)

    # If `X` is one-dimensional then expand it to two dimensions and save the
    # information, so that we can ensure the output is also one-dimensional
    onedim = len(X.shape) == 1
    if onedim:
        X = np.expand_dims(X, 0)

    # Get the full non-bootstrapped predictions of `X`
    preds = self._model(X) if callable(self._model) else self._model.predict(X)

    # If no quantiles should be outputted then simply return the predictions of the
    # underlying model
    if uncertainty is None and quantiles is None:
        return preds

    # Ensure that the underlying model has been fitted before predicting. This is only
    # a requirement if `uncertainty` is set, as we need access to `self.X_train`
    if not hasattr(self, "X_train") or self.X_train is None:
        raise RuntimeError(
            "This model has not been fitted yet! Call fit() "
            "before predicting new samples."
        )

    # Store the number of data points in the training and test datasets
    n_train = self.X_train.shape[0]
    n_test = X.shape[0]

    # The authors chose the number of bootstrap samples as the square root of the
    # number of samples in the training dataset
    if n_boots is None:
        n_boots = np.sqrt(n_train).astype(int)

    # Compute the m_i's and the validation residuals
    bootstrap_preds = np.empty((n_boots, n_test))
    for boot_idx in range(n_boots):
        train_idxs = rng.choice(range(n_train), size=n_train, replace=True)
        X_train = self.X_train[train_idxs, :]
        y_train = self.y_train[train_idxs]

        bootstrap_pred = self._model_fit_predict(X_train, y_train, X)
        bootstrap_preds[boot_idx] = bootstrap_pred

    # Centre the bootstrapped predictions across the bootstrap dimension
    bootstrap_preds -= np.mean(bootstrap_preds, axis=0)

    # Add up the bootstrap predictions and the hybrid train/val residuals
    C = np.array([m + o for m in bootstrap_preds for o in self.residuals])

    # Calculate the desired quantiles
    if quantiles is None and uncertainty is not None:
        quantiles = [uncertainty / 2, 1 - uncertainty / 2]
    quantile_vals = np.transpose(np.quantile(C, q=quantiles or [], axis=0))

    # Return the predictions and the desired quantiles
    if onedim:
        return preds[0], (preds + quantile_vals)[0]
    else:
        return preds, np.expand_dims(preds, axis=1) + quantile_vals


def fit(self, X: np.ndarray, y: np.ndarray, n_boots: Optional[int] = None):
    """Fits the model to the data.

    Args:
        X (float array):
            The array containing the data set, either of shape (f,) or (n, f), with n
            being the number of samples and f being the number of features.
        y (float array):
            The array containing the target values, of shape (n,)
        n_boots (int or None):
            The number of resamples to bootstrap. If None then it is set to the square
            root of the data set. Defaults to None
    """
    # Initialise random number generator
    rng = np.random.default_rng(self.random_seed)

    # Set the number of data points in the dataset
    n = X.shape[0]

    # Set default value of `n_boots` if it is not set
    if n_boots is None:
        n_boots = np.sqrt(n).astype(int)

    # Ensure that `X` and `y` are Numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    # Store `X` and `y` for predictions
    self.X_train = X
    self.y_train = y

    # Fit the underlying model and get predictions on the training dataset
    self._model.fit(X, y)
    preds = self._model(X) if callable(self._model) else self._model.predict(X)

    # Calculate the training residuals and aggregate them into quantiles, to enable
    # comparison with the validation residuals
    train_residuals = np.quantile(y - preds, q=np.arange(0, 1, 0.01))

    # Compute the m_i's and the validation residuals
    val_residuals_list = []
    for _ in range(n_boots):
        train_idxs = rng.choice(range(n), size=n, replace=True)
        val_idxs = [idx for idx in range(n) if idx not in train_idxs]

        X_train = X[train_idxs, :]
        y_train = y[train_idxs]
        X_val = X[val_idxs, :]
        y_val = y[val_idxs]

        boot_preds = self._model_fit_predict(X_train, y_train, X_val)
        val_residuals_list.append(y_val - boot_preds)

    # Aggregate the validation residuals into quantiles, to enable comparison with the
    # training residuals
    val_residuals = np.concatenate(val_residuals_list)
    val_residuals = np.quantile(val_residuals, q=np.arange(0, 1, 0.01))

    # Compute the no-information value
    permuted = rng.permutation(y) - rng.permutation(preds)
    no_info_error = np.mean(np.abs(permuted))
    no_info_val = np.abs(no_info_error - train_residuals)

    # Compute the .632+ bootstrap estimate for the sample noise and bias
    generalisation = np.abs(val_residuals.mean() - train_residuals.mean())
    relative_overfitting_rate = np.mean(generalisation / no_info_val)
    weight = 0.632 / (1 - 0.368 * relative_overfitting_rate)
    self.residuals = (1 - weight) * train_residuals + weight * val_residuals

    return self
