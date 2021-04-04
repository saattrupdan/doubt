'''Base class for data sets'''

import warnings
import requests
import abc
import re
from typing import Optional, Iterable, Tuple

import numpy as np
import pandas as pd


BASE_DATASET_DESCRIPTION = '''
    Parameters:
        cache (str or None, optional):
            The name of the cache. It will be saved to `cache` in the
            current working directory. If None then no cache will be saved.
            Defaults to '.dataset_cache'.

    Attributes:
        shape (tuple of integers):
            Dimensions of the data set
        columns (list of strings):
            List of column names in the data set

    Class attributes:
        url (string):
            The url where the raw data files can be downloaded
        feats (iterable):
            The column indices of the feature variables
        trgts (iterable):
            The column indices of the target variables
'''


class BaseDataset(object, metaclass=abc.ABCMeta):

    url: str
    feats: Iterable
    trgts: Iterable

    def __init__(self, cache: Optional[str] = '.dataset_cache'):
        self.cache = cache
        self._data = self.get_data()
        self.shape = self._data.shape
        self.columns = self._data.columns

    @abc.abstractmethod
    def _prep_data(self, data: bytes) -> pd.DataFrame:
        return

    def get_data(self) -> pd.DataFrame:
        ''' Download and prepare the dataset.

        Returns:
            Pandas DataFrame: The dataset.
        '''

        # Get name of dataset, being the class name converted to snake case
        name = re.sub(r'([A-Z])', r'_\1', type(self).__name__)
        name = name.lower().strip('_')

        try:
            if self.cache is not None:
                data = pd.read_hdf(self.cache, name)
        except (FileNotFoundError, KeyError):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                response = requests.get(self.url, verify=False)
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

    def split(self,
              test_size: Optional[float] = None,
              random_seed: Optional[float] = None) -> Tuple[np.ndarray]:
        '''Split dataset into features and targets and optionally also train/test.

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
        '''
        nrows = len(self._data)
        feats = type(self).feats
        trgts = type(self).trgts

        if test_size is not None:
            if random_seed is not None:
                np.random.seed(random_seed)
            test_idxs = np.random.random(size=(nrows,)) < test_size
            train_idxs = ~test_idxs

            X_train = self._data.iloc[train_idxs, feats].values
            y_train = self._data.iloc[train_idxs, trgts].values.squeeze()
            X_test = self._data.iloc[test_idxs, feats].values
            y_test = self._data.iloc[test_idxs, trgts].values.squeeze()

            return X_train, X_test, y_train, y_test

        else:
            X = self._data.iloc[:, feats].values
            y = self._data.iloc[:, trgts].values.squeeze()
            return X, y
