''' Base class for data sets '''

from pathlib import Path
import warnings
import requests
import abc
import re

import numpy as np
import pandas as pd

from typing import Optional
from typing import Iterable
from typing import Tuple

class BaseDataset(object, metaclass = abc.ABCMeta):

    url: str
    feats: Iterable
    trgts: Iterable

    def __init__(self, use_cache: bool = True, cache_name: str = '.cache'):
        self.cache = pd.HDFStore(f'{cache_name}.h5') if use_cache else {}
        self.data = self.get_data()
        self.shape = self.data.shape
        self.columns = self.data.columns

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
            data = self.cache[name]
        except KeyError:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                response = requests.get(self.url, verify = False)
            data = self._prep_data(response.content)
            if self.cache != {}:
                data.to_hdf(self.cache, name)
        return data

    def to_pandas(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def head(self):
        return self.data.head()

    def close(self):
        if self.cache != {}:
            self.cache.close()
        del self

    def __exit__(self):
        self.close()

    def __str__(self):
        return str(self.data)

    def split(self, test_size: Optional[float] = None) -> Tuple[np.ndarray]:
        ''' 
        Split dataset into features and targets and optionally also train/test.

        Args:
            test_size (float or None):
                The fraction of the dataset that will constitute the test
                set. If None then no train/test split will happen. Defaults
                to None.
            random_seed (float or None):
                The random seed used for the train/test split. If None then
                a random number will be chosen. Defaults to None.

        Returns:
            If ``test_size`` is not `None` then a tuple of numpy arrays
            (X_train, y_train, X_test, y_test) is returned, and otherwise 
            the tuple (X, y) is returned.
        '''
        nrows = len(self.data)
        feats = type(self).feats
        trgts = type(self).trgts

        if test_size is not None:
            test_idxs = np.random.choice(nrows, p = test_size, size = (nrows,))
            train_idxs = np.isin(test_idxs, np.arange(nrows), invert = True)

            X_train = self.data.iloc[train_idxs, feats]
            y_train = self.data.iloc[train_idxs, trgts]
            X_test = self.data.iloc[test_idxs, feats]
            y_test = self.data.iloc[test_idxs, trgts]
            
            return X_train, y_train, X_test, y_test

        else:
            X = self.data.iloc[:, feats].values
            y = self.data.iloc[:, trgts].values
            return X, y
