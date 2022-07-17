"""Base class for data sets"""

import re
import warnings
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

BASE_DATASET_DESCRIPTION = """
    Parameters:
        cache (str or None, optional):
            The name of the cache. It will be saved to `cache` in the
            current working directory. If None then no cache will be saved.
            Defaults to '.dataset_cache'.

    Attributes:
        cache (str or None):
            The name of the cache.
        shape (tuple of integers):
            Dimensions of the data set
        columns (list of strings):
            List of column names in the data set
"""


class BaseDataset(ABC):

    _url: str
    _features: Iterable
    _targets: Iterable

    def __init__(self, cache: Optional[str] = ".dataset_cache"):
        self.cache = cache
        self._data = self.get_data()
        self.shape = self._data.shape
        self.columns = self._data.columns

    @abstractmethod
    def _prep_data(self, data: bytes) -> pd.DataFrame:
        return

    def get_data(self) -> pd.DataFrame:
        """Download and prepare the dataset.

        Returns:
            Pandas DataFrame: The dataset.
        """

        # Get name of dataset, being the class name converted to snake case
        name = re.sub(r"([A-Z])", r"_\1", type(self).__name__)
        name = name.lower().strip("_")

        try:
            if self.cache is not None:
                data = pd.read_hdf(self.cache, name)
        except (FileNotFoundError, KeyError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                response = requests.get(self._url, verify=False)
            data = self._prep_data(response.content)
            if self.cache is not None:
                data.to_hdf(self.cache, name)
        return data

    def to_pandas(self) -> pd.DataFrame:
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._data.head(n)

    def close(self):
        del self._data
        del self

    def __exit__(self, exc_type: str, exc_value: str, exc_traceback: str):
        self.close()

    def __str__(self) -> str:
        return str(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def _repr_html_(self):
        return self._data._repr_html_()

    def split(
        self, test_size: Optional[float] = None, random_seed: Optional[float] = None
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Split dataset into features and targets and optionally train/test.

        Args:
            test_size (float or None):
                The fraction of the dataset that will constitute the test
                set. If None then no train/test split will happen. Defaults
                to None.
            random_seed (float or None):
                The random seed used for the train/test split. If None then
                a random number will be chosen. Defaults to None.

        Returns:
            If `test_size` is not `None` then a tuple of numpy arrays
            (X_train, y_train, X_test, y_test) is returned, and otherwise
            the tuple (X, y) of numpy arrays is returned.
        """
        # Initialise random number generator
        rng = np.random.default_rng(random_seed)

        nrows = len(self._data)
        features = self._features
        targets = self._targets

        if test_size is not None:
            test_idxs = rng.random(size=(nrows,)) < test_size
            train_idxs = ~test_idxs

            X_train = self._data.iloc[train_idxs, features].values
            y_train = self._data.iloc[train_idxs, targets].values.squeeze()
            X_test = self._data.iloc[test_idxs, features].values
            y_test = self._data.iloc[test_idxs, targets].values.squeeze()

            return X_train, X_test, y_train, y_test

        else:
            X = self._data.iloc[:, features].values
            y = self._data.iloc[:, targets].values.squeeze()
            return X, y
