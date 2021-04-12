'''Solar flare data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import io


class SolarFlare(BaseDataset):
    __doc__ = f'''
    Each class attribute counts the number of solar flares of a certain class
    that occur in a 24 hour period.

    The database contains 3 potential classes, one for the number of times a
    certain type of solar flare occured in a 24 hour period.

    Each instance represents captured features for 1 active region on the sun.

    The data are divided into two sections. The second section (flare.data2)
    has had much more error correction applied to the it, and has consequently
    been treated as more reliable.

    {BASE_DATASET_DESCRIPTION}

    Features:
        class (int):
            Code for class (modified Zurich class). Ranges from 0 to 6
            inclusive
        spot_size (int):
            Code for largest spot size. Ranges from 0 to 5 inclusive
        spot_distr (int):
            Code for spot distribution. Ranges from 0 to 3 inclusive
        activity (int):
            Binary feature indicating 1 = reduced and 2 = unchanged
        evolution (int):
            0 = decay, 1 = no growth and 2 = growth
        flare_activity (int):
            Previous 24 hour flare activity code, where 0 = nothing as big
            as an M1, 1 = one M1 and 2 = more activity than one M1
        is_complex (int):
            Binary feature indicating historically complex
        became_complex (int):
            Binary feature indicating whether the region became historically
            complex on this pass across the sun's disk
        large (int):
            Binary feature, indicating whether area is large
        large_spot (int):
            Binary feature, indicating whether the area of the largest
            spot is greater than 5

    Targets:
        C-class (int):
            C-class flares production by this region in the following 24
            hours (common flares)
        M-class (int):
            M-class flares production by this region in the following 24
            hours (common flares)
        X-class (int):
            X-class flares production by this region in the following 24
            hours (common flares)

    Source:
        https://archive.ics.uci.edu/ml/datasets/Solar+Flare

    Examples:
        Load in the data set::

            >>> dataset = SolarFlare()
            >>> dataset.shape
            (1066, 13)

        Split the data set into features and targets, as NumPy arrays::

            >>> X, y = dataset.split()
            >>> X.shape, y.shape
            ((1066, 10), (1066, 3))

        Perform a train/test split, also outputting NumPy arrays::

            >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
            >>> X_train, X_test, y_train, y_test = train_test_split
            >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
            ((831, 10), (831, 3), (235, 10), (235, 3))

        Output the underlying Pandas DataFrame::

            >>> df = dataset.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
    '''

    _url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
            'solar-flare/flare.data2')

    _features = range(10)
    _targets = range(10, 13)

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        # Convert the bytes into a file-like object
        csv_file = io.BytesIO(data)

        # Load in dataframe
        cols = ['class', 'spot_size', 'spot_distr', 'activity', 'evolution',
                'flare_activity', 'is_complex', 'became_complex', 'large',
                'large_spot', 'C-class', 'M-class', 'X-class']
        df = pd.read_csv(csv_file, sep=' ', skiprows=[0], names=cols)

        # Encode class
        encodings = ['A', 'B', 'C', 'D', 'E', 'F', 'H']
        df['class'] = df['class'].map(lambda x: encodings.index(x))

        # Encode spot size
        encodings = ['X', 'R', 'S', 'A', 'H', 'K']
        df['spot_size'] = df.spot_size.map(lambda x: encodings.index(x))

        # Encode spot distribution
        encodings = ['X', 'O', 'I', 'C']
        df['spot_distr'] = df.spot_distr.map(lambda x: encodings.index(x))

        return df
