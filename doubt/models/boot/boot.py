''' Bootstrap wrapper for datasets and models '''

#TODO: Add documentation to methods, and sources for the bootstrap methods

from typing import Optional
from typing import Union
from typing import Sequence
from typing import Callable

import numpy as np

FloatArray = Sequence[float]
NumericArray = Sequence[Union[float, int]]

class Boot(object):
    ''' Bootstrap wrapper for datasets and models.

    Datasets can be any sequence of numeric input, from which bootstrapped
    statistics can be calculated, with confidence intervals included.

    The models can be any model with a `predict` method, such as all 
    the models in `scikit-learn`, and the bootstrapped model can then
    produce predictions with prediction intervals.

    Note:
        For PyTorch and TensorFlow models it is recommended to use `TorchBoot`
        and `TFBoot` instead, to have full integration with these frameworks.

    Args:
        input (array-like or model):
            Either a dataset to calculate bootstrapped statistics on, or an
            model for which bootstrapped predictions will be computed.

    Methods:
        compute_statistic(statistic, n_boots, agg) -> float or array of floats
        predict(array, q) -> float or array of floats
        
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

    '''
    def __init__(self, input, random_seed: Optional[float] = None):
        self.random_seed = random_seed
        fn = getattr(input, 'predict', None)
        data = getattr(input, '__getitem__', None)

        if fn is not None and callable(fn):
            self.mode = 'model'
            self.model = input

        elif data is not None:
            self.X_train = None
            self.y_train = None
            self.mode = 'data'
            self.data = np.asarray(input)

        else:
            raise RuntimeError('Input not recognised.')

    def compute_statistic(self,
        statistic: Callable[[NumericArray], float], 
        n_boots: Optional[int] = None,
        uncertainty: float = .05,
        return_all: bool = False) -> Union[float, FloatArray]:
        ''' Compute bootstrapped statistic. '''

        if not self.mode == 'data':
            raise RuntimeError('This Boot is not set up for computing '\
                               'statistics on data. Initialise with a '\
                               'dataset instead.')
        
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

    def predict(self, X, 
        n_boots: Optional[int] = None,
        uncertainty: float = .05) -> Union[float, FloatArray]:
        ''' Compute bootstrapped predictions. '''

        if not self.mode == 'model':
            raise RuntimeError('This Boot is not set up for predictions. '\
                               'Initialise with a model instead.')

        if self.random_seed is not None: np.random.seed(self.random_seed)

        X = np.asarray(X)
        n = self.X_train.shape[0]

        twodim = (len(X.shape) == 1)
        if twodim: X = np.expand_dims(X, 0)

        # The authors choose the number of bootstrap samples as the square 
        # root of the number of samples
        if n_boots is None: n_boots = np.sqrt(n).astype(int)

        # Compute the m_i's and the validation residuals
        bootstrap_preds = np.empty(n_boots)
        val_residuals = []
        for b in range(n_boots):
            train_idxs = np.random.choice(range(n), size = n, replace = True)
            val_idxs = [idx for idx in range(n) if idx not in train_idxs]

            X_train = self.X_train[train_idxs, :]
            y_train = self.y_train[train_idxs]
            self.model.fit(X_train, y_train)

            X_val = self.X_train[val_idxs, :]
            y_val = self.y_train[val_idxs]
            preds = self.model.predict(X_val)

            val_residuals.append(y_val - preds)
            bootstrap_preds[b] = self.model.predict(X)
        bootstrap_preds -= np.mean(bootstrap_preds)
        val_residuals = np.concatenate(val_residuals)
        val_residuals = np.quantile(val_residuals, q = np.arange(0, 1, .01))

        # Compute the .632+ bootstrap estimate for the sample noise and bias
        generalisation = np.abs(val_residuals - self.train_residuals)
        relative_overfitting_rate = np.mean(generalisation / self.no_info_val)
        weight = .632 / (1 - .368 * relative_overfitting_rate)
        residuals = (1 - weight) * self.train_residuals + weight * val_residuals

        # Construct the C set and get the quantiles
        C = np.array([m + o for m in bootstrap_preds for o in residuals])
        quantiles = np.quantile(C, q = [uncertainty / 2, 1 - uncertainty / 2])

        preds = self.model.predict(X)

        if twodim:
            return preds[0], (preds + quantiles)[0]
        else:
            return preds, preds + quantiles

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.X_train = X
        self.y_train = y

        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.train_residuals = np.quantile(y - preds, q = np.arange(0, 1, .01))

        permuted = np.random.permutation(y) - np.random.permutation(preds)
        no_info_error = np.mean(np.abs(permuted))
        self.no_info_val = np.abs(no_info_error - self.train_residuals)
        return self
