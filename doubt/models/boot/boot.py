''' Bootstrap wrapper for datasets and models '''

from typing import Optional
from typing import Union
from typing import Sequence
from typing import Callable

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
        >>> import numpy as np
        >>> from doubt.datasets import FishToxicity
        >>> data = FishToxicity.response()
        >>> boot = Boot(data)
        >>> boot.compute_statistic(np.mean, n_boots = 10, uncertainty = .05)
        x, (a, b)

        Alternatively, we can output the whole bootstrap distribution:
        >>> boot.compute_statistic(np.mean, n_boots = 10, return_all = True)
        x, (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)

        Wrap a scikit-learn model and get prediction intervals:
        >>> from sklearn.linear_model import LinearRegression
        >>> from doubt.datasets import PowerPlant
        >>> data = PowerPlant()
        >>> linreg = Boot(LinearRegression())
        >>> linreg.fit(data)
        >>> linreg.predict((10, 30, 1000, 50), uncertainty = .05)
        x, (a, b)

    '''
    def __init__(self, input):
        fn = getattr(input, 'predict', None)
        data = getattr(input, '__getitem__', None)

        if fn is not None and callable(fn):
            self.mode = 'model'
            self.model = input

        elif data is not None and callable(fn):
            self.mode = 'data'
            self.data = input

        else:
            raise RuntimeError('Input not recognised.')

    def compute_statistic(
        statistic: Callable[[NumericArray], float], 
        n_boots: int,
        agg: Callable[[NumericArray], float]) -> Union[float, FloatArray]:
        ''' Compute bootstrapped statistic. '''

        if not self.mode == 'data':
            raise RuntimeError('This Boot is not set up for computing '\
                               'statistics on data. Initialise with a '\
                               'dataset instead.')

        raise NotImplementedError

    def predict(self, *args, q: Optional[FloatArray] = None, 
                **kwargs) -> Union[float, FloatArray]:
        ''' Compute bootstrapped predictions. '''

        if not self.mode == 'model':
            raise RuntimeError('This Boot is not set up for predictions. '\
                               'Initialise with a model instead.')

        preds = self.model.predict(*args)
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
