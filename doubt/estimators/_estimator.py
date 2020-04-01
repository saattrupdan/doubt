''' Base class for estimators '''

import abc
from typing import Sequence, Union

FloatMatrix = Sequence[Sequence[float]]
FloatNDArray = Union[Sequence[float], FloatMatrix]

class BaseEstimator(object, metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, X: FloatMatrix, **kwargs) -> FloatNDArray:
        return 

    @abc.abstractmethod
    def fit(self, X: FloatMatrix, y: FloatNDArray, **kwargs):
        return 

    def plot_pred_interval(self, X: FloatMatrix, y: FloatNDArray, **kwargs):
        return 

    def __call__(self, X: FloatMatrix, **kwargs) -> FloatNDArray:
        return self.predict(X)
