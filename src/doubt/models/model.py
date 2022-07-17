"""Base class for estimators."""

from typing import Protocol, Tuple, Union

import numpy as np


class Model(Protocol):
    def __init__(self, *args, **kwargs):
        ...

    def predict(
        self, X: np.ndarray, **kwargs
    ) -> Tuple[Union[float, np.ndarray], np.ndarray]:
        ...

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        ...

    def __call__(
        self, X: np.ndarray, **kwargs
    ) -> Tuple[Union[float, np.ndarray], np.ndarray]:
        ...
