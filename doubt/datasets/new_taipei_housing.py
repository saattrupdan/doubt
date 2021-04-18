'''New Taipei Housing data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import io


class NewTaipeiHousing(BaseDataset):
    __doc__ = f'''
    The "real estate valuation" is a regression problem. The market historical
    data set of real estate valuation are collected from Sindian Dist., New
    Taipei City, Taiwan.

    {BASE_DATASET_DESCRIPTION}

    Features:
        transaction_date (float):
            The transaction date encoded as a floating point value. For
            instance, 2013.250 is March 2013 and 2013.500 is June March
        house_age (float):
            The age of the house
        mrt_distance (float):
            Distance to the nearest MRT station
        n_stores (int):
            Number of convenience stores
        lat (float):
            Latitude
        lng (float):
            Longitude

    Targets:
        house_price (float):
            House price of unit area

    Source:
        https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set

    Examples:
        Load in the data set::

            >>> dataset = NewTaipeiHousing()
            >>> dataset.shape
            (414, 7)

        Split the data set into features and targets, as NumPy arrays::

            >>> X, y = dataset.split()
            >>> X.shape, y.shape
            ((414, 6), (414,))

        Perform a train/test split, also outputting NumPy arrays::

            >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
            >>> X_train, X_test, y_train, y_test = train_test_split
            >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
            ((323, 6), (323,), (91, 6), (91,))

        Output the underlying Pandas DataFrame::

            >>> df = dataset.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
    '''

    _url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
            '00477/Real%20estate%20valuation%20data%20set.xlsx')

    _features = range(6)
    _targets = [6]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        # Convert the bytes into a file-like object
        xlsx_file = io.BytesIO(data)

        # Load in the dataframe
        cols = ['idx', 'transaction_date', 'house_age', 'mrt_distance',
                'n_stores', 'lat', 'lng', 'house_price']
        df = pd.read_excel(xlsx_file, header=0, names=cols)

        # Remove the index
        df = df.iloc[:, 1:]

        return df
