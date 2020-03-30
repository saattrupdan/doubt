import requests
import pandas as pd
from typing import Optional, Union
import abc
from pathlib import Path
import re

class BaseDataset(object, metaclass = abc.ABCMeta):

    url: str

    def __init__(self, use_cache: bool = True):
        name = re.sub(r'([A-Z])', r'_\1', type(self).__name__)
        name = name.lower().strip('_')
        self.cache = pd.HDFStore('.cache.h5') if use_cache else {}
        self.data = self.get_data(name)
        self.shape = self.data.shape

    @abc.abstractmethod
    def prep_data(self, data: str):
        return 

    def get_data(self, name: str):
        try:
            data = self.cache[name]
        except KeyError:
            response = requests.get(self.url)
            data = self.prep_data(response.content)
            if self.cache:
                data.to_hdf(self.cache, name)
        return data

    def __len__(self):
        return len(self.data)

    def close(self):
        self.data.close()
        del self

    def __exit__(self):
        self.close()

    def __str__(self):
        return str(self.data)

    def split(self, 
        train_size: Optional[float] = None, 
        test_size: Optional[float] = None):
        
        if train_size is None and test_size is None:
            raise TypeError('One of `train_size` and `test_size` must '\
                            'be specified.')
        
        raise NotImplementedError
