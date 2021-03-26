'''Bootstrap wrapper for datasets and models'''

from typing import Optional, Union, Tuple, Sequence, Callable
import numpy as np
from types import MethodType

FloatArray = Sequence[float]
NumericArray = Sequence[Union[float, int]]


class Boot:
    '''Bootstrap wrapper for datasets and models.

    Datasets can be any sequence of numeric input, from which bootstrapped
    statistics can be calculated, with confidence intervals included.

    The models can be any model that is either callable or equipped with
    a `predict` method, such as all the models in `scikit-learn`, `pytorch`
    and `tensorflow`, and the bootstrapped model can then produce predictions
    with prediction intervals.

    The bootstrapped confidence intervals are computed using the .632+ estimate
    from [1], with the caveat that the no-information error rate that is used
    here is an approximation, to avoid n^2 running time. The bootstrapped
    prediction intervals are computed using the method from [2].

    Args:
        input (float array or model):
            Either a dataset to calculate bootstrapped statistics on, or an
            model for which bootstrapped predictions will be computed.
        random_seed (float or None):
            The random seed used for bootstrapping. If set to None then no
            seed will be set. Defaults to None.

    Methods:
        compute_statistic(statistic, n_boots, agg) -> tuple
        predict(X, n_boots, uncertainty) -> tuple
        fit(X, y) -> self

    Examples:
        Compute the bootstrap distribution of the mean, with a 95% confidence
        interval:
        >>> from doubt.datasets import FishToxicity
        >>> X, y = FishToxicity().split()
        >>> boot = Boot(y, random_seed = 42)
        >>> boot.compute_statistic(np.mean)
        (4.064430616740088, array([3.99133279, 4.15605735]))

        Alternatively, we can output the whole bootstrap distribution:
        >>> boot.compute_statistic(np.mean, n_boots = 3, return_all = True)
        (4.064430616740088, array([4.10546476, 4.02547137, 4.03936894]))

        Wrap a scikit-learn model and get prediction intervals:
        >>> from sklearn.linear_model import LinearRegression
        >>> from doubt.datasets import PowerPlant
        >>> X, y = PowerPlant().split()
        >>> linreg = Boot(LinearRegression())
        >>> linreg = linreg.fit(X, y)
        >>> linreg.predict([10, 30, 1000, 50])
        (array([481.92031021]), array([473.43314309, 490.0313962 ]))

    Sources:
        [1]: Friedman, J., Hastie, T., & Tibshirani, R. (2001). The elements
             of statistical learning (Vol. 1, No. 10). New York: Springer
             series in statistics.
        [2]: Kumar, S., & Srivistava, A. N. (2012). Bootstrap prediction
             intervals in non-parametric regression with applications to
             anomaly detection.
    '''
    def __init__(self, input: object, random_seed: Optional[float] = None):
        self.random_seed = random_seed

        # Input is a model
        if callable(input) or hasattr(input, 'predict'):
            self.mode = 'model'
            self.model_fit = input.fit
            self.model_predict = input if callable(input) else input.predict
            self.fit = MethodType(fit, self)
            self.predict = MethodType(predict, self)

        # Input is a dataset
        elif hasattr(input, '__getitem__'):
            self.data = np.asarray(input)
            self.compute_statistic = MethodType(compute_statistic, self)

        else:
            raise TypeError('Input not recognised.')

def compute_statistic(self,
    statistic: Callable[[NumericArray], float],
    n_boots: Optional[int] = None,
    uncertainty: float = .05,
    return_all: bool = False) -> Tuple[float, FloatArray]:
    '''Compute bootstrapped statistic.

    Args:
        statistic (numeric array -> float):
            The statistic to be computed on bootstrapped samples.
        n_boots (int or None):
            The number of resamples to bootstrap. If None then it is set
            to the square root of the data set. Defaults to None
        uncertainty (float):
            The uncertainty used to compute the confidence interval
            of the bootstrapped statistic. Not used if `return_all` is
            set to True. Defaults to 0.05.
        return_all (bool):
            Whether all bootstrapped statistics should be returned instead
            of the confidence interval. Defaults to False.

    Returns:
        pair of a float and an array of floats:
            The bootstrapped statistic and either the confidence interval
            or all of the bootstrapped statistics
    '''
    if self.random_seed is not None: np.random.seed(self.random_seed)

    n = self.data.shape[0]

    if n_boots is None: n_boots = np.sqrt(n).astype(int)

    statistics = np.empty((n_boots,), dtype = float)
    for b in range(n_boots):
        boot_idxs = np.random.choice(range(n), size = n, replace = True)
        statistics[b] = statistic(self.data[boot_idxs])

    if return_all:
        return statistic(self.data), statistics
    else:
        lower = uncertainty / 2
        upper = 1. - lower
        interval = np.quantile(statistics, q = (lower, upper))
        return statistic(self.data), interval

def predict(self,
    X: FloatArray,
    n_boots: Optional[int] = None,
    uncertainty: float = .05) -> Tuple[Union[float, FloatArray], FloatArray]:
    '''Compute bootstrapped predictions.

    This is an extension of the prediction method calculation in [2], which
    also takes validation error into account. To remedy this, the .632+
    bootstrap estimate from [3] has been used. Read more in [1].

    Args:
        X (float array):
            The array containing the data set, either of shape (f,)
            or (n, f), with n being the number of samples and f being
            the number of features.
        n_boots (int or None):
            The number of resamples to bootstrap. If None then it is set
            to the square root of the data set. Defaults to None
        uncertainty (float):
            The uncertainty used to compute the confidence interval
            of the bootstrapped statistic. Not used if `return_all` is
            set to True. Defaults to 0.05.

    Returns:
        pair of float arrays:
            The bootstrapped predictions and the confidence intervals

    References:
        [1]: https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/
        [2]: https://ntrs.nasa.gov/api/citations/20130014367/downloads/20130014367.pdf
        [3]: https://web.stanford.edu/~hastie/ElemStatLearn/
    '''
    if not hasattr(self, 'X_train') or self.X_train is None:
        raise RuntimeError('This model has not been fitted yet! Call fit() '
                            'before predicting new samples.')

    if self.random_seed is not None: np.random.seed(self.random_seed)

    X = np.asarray(X)
    n = self.X_train.shape[0]

    onedim = (len(X.shape) == 1)
    if onedim: X = np.expand_dims(X, 0)

    # The authors choose the number of bootstrap samples as the square
    # root of the number of samples
    if n_boots is None: n_boots = np.sqrt(n).astype(int)

    # Compute the m_i's and the validation residuals
    bootstrap_preds = []
    val_residuals = []
    for _ in range(n_boots):
        train_idxs = np.random.choice(range(n), size = n, replace = True)
        val_idxs = [idx for idx in range(n) if idx not in train_idxs]

        X_train = self.X_train[train_idxs, :]
        y_train = self.y_train[train_idxs]
        self.model_fit(X_train, y_train)

        X_val = self.X_train[val_idxs, :]
        y_val = self.y_train[val_idxs]
        preds = self.model_predict(X_val)

        val_residuals.append(y_val - preds)
        bootstrap_preds.append(self.model_predict(X))

    bootstrap_preds = np.stack(bootstrap_preds, axis=0)
    bootstrap_preds -= np.mean(bootstrap_preds, axis=0)
    val_residuals = np.concatenate(val_residuals)
    val_residuals = np.quantile(val_residuals, q=np.arange(0, 1, .01))

    # Compute the .632+ bootstrap estimate for the sample noise and bias
    generalisation = np.abs(val_residuals - self.train_residuals)
    relative_overfitting_rate = np.mean(generalisation / self.no_info_val)
    weight = .632 / (1 - .368 * relative_overfitting_rate)
    residuals = (1 - weight) * self.train_residuals + weight * val_residuals

    # Construct the C set and get the quantiles
    C = np.array([m + o for m in bootstrap_preds for o in residuals])
    quantiles = np.quantile(C, q=[uncertainty / 2, 1 - uncertainty / 2], axis=0)
    quantiles = np.transpose(quantiles)

    preds = self.model_predict(X)

    if onedim:
        return preds[0], (preds + quantiles)[0]
    else:
        return preds, np.expand_dims(preds, axis=1) + quantiles

def fit(self, X: FloatArray, y: FloatArray):
    '''Fits the model to the data.

    Args:
        X (float array):
            The feature matrix.
        y (float array):
            The target matrix.
    '''
    X = np.asarray(X)
    y = np.asarray(y)
    self.X_train = X
    self.y_train = y

    self.model_fit(X, y)
    preds = self.model_predict(X)
    self.train_residuals = np.quantile(y - preds, q = np.arange(0, 1, .01))

    permuted = np.random.permutation(y) - np.random.permutation(preds)
    no_info_error = np.mean(np.abs(permuted))
    self.no_info_val = np.abs(no_info_error - self.train_residuals)
    return self
