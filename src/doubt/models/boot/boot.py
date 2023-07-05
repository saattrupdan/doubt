"""Bootstrap wrapper for datasets and models"""

import copy
import multiprocessing as mp
from types import MethodType
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray


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

            >>> from doubt.datasets import FishToxicity, PowerPlant
            >>> import numpy as np
            >>> X, y = FishToxicity().split()
            >>> boot = Boot(y, random_seed=42)
            >>> preds, intervals = boot.compute_statistic(np.mean)
            >>> round(preds, 2)
            4.06
            >>> np.around(intervals, 2)
            array([3.98, 4.17])

        Alternatively, we can output the whole bootstrap distribution::

            >>> preds, intervals = boot.compute_statistic(
            ...     np.mean, n_boots=3, return_all=True
            ... )
            >>> round(preds, 2)
            4.06
            >>> np.around(intervals, 2)
            array([4.06, 4.06, 4.06])

        Wrap a scikit-learn model and get prediction intervals::

            >>> from sklearn.linear_model import LinearRegression
            >>> X, y = PowerPlant().split()
            >>> linreg = Boot(LinearRegression(), random_seed=42)
            >>> linreg = linreg.fit(X, y)
            >>> preds, intervals = linreg.predict([10, 30, 1000, 50], uncertainty=0.05)
            >>> round(preds, 2)
            482.0
            >>> np.around(intervals, 2)
            array([473.5 , 490.14])

        Instead of specifying the uncertainty, we can also specify the precise
        quantiles that we are interested in::

            >>> preds, intervals = linreg.predict(
            ...     [10, 30, 1000, 50], quantiles=[0.1, 0.5]
            ... )
            >>> np.around(intervals, 2)
            array([476.45, 481.82])

        Lastly, as with the `compute_statistic` method, we can also output the whole
        bootstrap distribution with the `return_all` argument::

            >>> preds, bootstrap_samples = linreg.predict(
            ...     [10, 30, 1000, 50], return_all=True
            ... )
            >>> np.around(bootstrap_samples, 2)
            array([-43.7 , -10.02,  -8.63, ...,   8.28,   9.06,  10.19])

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
    model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """Fit the underlying model and perform predictions with it.

    This requires `self._model` to be set and that it is either callable or have a
    `predict` method.

    Args:
        model (object with `fit` and `predict` methods):
            The model to fit and predict with.
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
    model = copy.deepcopy(model)
    model.fit(X_train, y_train)
    if callable(model):
        return model(X_test)
    else:
        return model.predict(X_test)

def _model_fit(
    model, X_train: np.ndarray, y_train: np.ndarray
):
    """Fit the underlying model.

    Args:
        model (object with `fit` method):
            The model to fit with.
        X_train (float matrix):
            Feature matrix for training, of shape
            (n_train_samples, n_features).
        y_train (float array):
            Target array, of shape (n_train_samples,).

    Returns:
        model (object with `fit` and `predict` methods):
            The model to fit
    """
    model = copy.deepcopy(model)
    model.fit(X_train, y_train)
    return model

def _model_predict(
    model, X: np.ndarray
) -> np.ndarray:
    """Perform predictions with the underlying model.

    This requires that the model is either callable or have a `predict` method.

    Args:
        model (object with `fit` and `predict` methods):
            The model to predict with.
        X (float matrix):
            Feature matrix for predicting, of shape
            (n_test_samples, n_features).

    Returns:
        Numpy array:
            Predictions, of shape (n_test_samples,)
    """
    if callable(model):
        return model(X)
    else:
        return model.predict(X)

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
    quantiles: Optional[Union[np.ndarray, List[float]]] = None,
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
    n = int(self.data.shape[0])

    # Set default value of the number of bootstrap samples if `n_boots` is not set
    if n_boots is None:
        n_boots = int(np.sqrt(n).astype(int))

    # Compute the bootstrapped statistics
    boot_idxs = rng.choice(n, size=(n_boots, n), replace=True)
    statistics = np.apply_along_axis(statistic, 1, self.data[boot_idxs])

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
    quantiles: Optional[Union[np.ndarray, List[float]]] = None,
    return_all: bool = False,
    n_jobs: int = None
) -> Union[Union[float, NDArray], Tuple[Union[float, NDArray], NDArray]]:
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
            prediction. For instance, an uncertainty of 0.05 will yield a 95%
            prediction interval. Will not be used if `quantiles` or `return_all` is
            used. Defaults to None.
        quantiles (sequence of floats or None, optional):
            List of quantiles to output. Will override the value of `uncertainty` if
            specified. Will not be used if `return_all` is True. Defaults to None.
        return_all (bool, optional):
            Whether the raw bootstrapped predictions should be returned. Will override
            the values of both `quantiles` and `uncertainty`. Defaults to False.
        n_jobs: (int or None):
            The number of jobs to use for parallelization. If None then it is equal to the
            number of available cpus. Defaults to None

    Returns:
        float array or pair of float arrays:
            The bootstrapped predictions, and the confidence intervals if `uncertainty`
            is not None, the specified quantiles if `quantiles` is not None, or the raw
            bootstrapped values if `return_all` is True.
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
    if uncertainty is None and quantiles is None and not return_all:
        return preds

    # Ensure that the underlying model has been fitted before predicting. This is only
    # a requirement if `uncertainty` is set, as we need access to `self.X_train`
    if not hasattr(self, "X_train") or self.X_train is None:
        raise RuntimeError(
            "This model has not been fitted yet! Call fit() before predicting new "
            "samples."
        )

    # Store the number of data points in the training and test datasets
    n_train = self.X_train.shape[0]

    # The authors chose the number of bootstrap samples as the square root of the
    # number of samples in the training dataset
    if n_boots is None:
        n_boots = int(np.sqrt(n_train).astype(int))

    # Sample the bootstrap indices
    train_idxs = rng.choice(n_train, size=(n_boots, n_train), replace=True)

    #set the number of jobs to use
    if n_jobs is None:
        jobs = mp.cpu_count() - 1
    else:
        jobs = n_jobs
    # Run the worker function in parallel
    with Parallel(n_jobs=jobs) as parallel:
        bootstrap_preds_list = parallel(
            delayed(_model_predict)(
                model=self._models[boot_idx],
                X=X,
            )
            for boot_idx in range(n_boots)
        )

    # Convert the list of predictions to a Numpy array
    bootstrap_preds = np.array(bootstrap_preds_list)

    # Centre the bootstrapped predictions across the bootstrap dimension
    bootstrap_preds -= np.mean(bootstrap_preds, axis=0)

    # Add up the bootstrap predictions and the hybrid train/val residuals
    C = np.array([m + o for m in bootstrap_preds for o in self.residuals])

    # If we are dealing with a single sample then we replace the predictions with the
    # first sample
    if onedim:
        preds = preds[0]

    # Next, we compute the associated uncertainty data, if requested. Firstly, if
    # `return_all` is set then we return the raw bootstrapped predictions, being the
    # `C` array
    if return_all:
        if onedim:
            return preds, C[:, 0]
        else:
            return preds, C.T

    # If the uncertainty has been specified and not the quantiles, then we compute
    # the quantiles from the uncertainty
    if quantiles is None and uncertainty is not None:
        quantiles = [uncertainty / 2, 1 - uncertainty / 2]

    # Compute the quantiles of the `C` array
    quantile_vals = np.transpose(np.quantile(C, q=quantiles or [], axis=0))

    # Add the quantiles to the predictions to get the final prediction intervals
    # as the uncertainty data
    if onedim:
        intervals = preds + quantile_vals[0]
    else:
        intervals = np.expand_dims(preds, axis=1) + quantile_vals

    # Return the predictions and the prediction intervals
    return preds, intervals


def fit(self, X: np.ndarray, y: np.ndarray, n_boots: Optional[int] = None, n_jobs: Optional[int] = None):
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
        n_jobs: (int or None):
            The number of jobs to use for parallelization. If None then it is equal to the
            number of available cpus minus one. Defaults to None
    """
    # Initialise random number generator
    rng = np.random.default_rng(self.random_seed)

    # Set the number of data points in the dataset
    n = X.shape[0]

    # Set default value of `n_boots` if it is not set
    if n_boots is None:
        n_boots = int(np.sqrt(n).astype(int))

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

    # Sample the bootstrap indices
    train_idxs = rng.choice(n, size=(n_boots, n), replace=True)

    # Set the number of jobs
    if n_jobs is None:
        jobs = mp.cpu_count() - 1
    else:
        jobs = n_jobs
    # Run the worker function in parallel
    with Parallel(n_jobs=jobs) as parallel:
        self._models = parallel(
            delayed(_model_fit)(
                model=self._model,
                X_train=self.X_train[train_idxs[boot_idx], :],
                y_train=self.y_train[train_idxs[boot_idx]],

            )
            for boot_idx in range(n_boots)
        )
    with Parallel(n_jobs=jobs) as parallel:
        bootstrap_preds = parallel(
            delayed(_model_predict)(
                model=self._models[boot_idx],
                X=X[[idx for idx in range(n) if idx not in train_idxs[boot_idx]]]
            )
            for boot_idx in range(n_boots)
        )

    # Compute the validation residuals
    val_residuals_list = [
        y[[idx for idx in range(n) if idx not in train_idxs[boot_idx]]]
        - bootstrap_preds[boot_idx]
        for boot_idx in range(n_boots)
    ]

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
