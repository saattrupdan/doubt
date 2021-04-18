'''Protein data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import io


class Protein(BaseDataset):
    __doc__ = f'''
    This is a data set of Physicochemical Properties of Protein Tertiary
    Structure. The data set is taken from CASP 5-9. There are 45730 decoys
    and size varying from 0 to 21 armstrong.

    {BASE_DATASET_DESCRIPTION}

    Features:
        F1 (float):
            Total surface area
        F2 (float):
            Non polar exposed area
        F3 (float):
            Fractional area of exposed non polar residue
        F4 (float):
            Fractional area of exposed non polar part of residue
        F5 (float):
            Molecular mass weighted exposed area
        F6 (float):
            Average deviation from standard exposed area of residue
        F7 (float):
            Euclidean distance
        F8 (float):
            Secondary structure penalty
        F9 (float):
            Spacial Distribution constraints (N,K Value)

    Targets:
        RMSD (float):
            Size of the residue

    Source:
        https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure

    Examples:
        Load in the data set::

            >>> dataset = Protein()
            >>> dataset.shape
            (45730, 10)

        Split the data set into features and targets, as NumPy arrays::

            >>> X, y = dataset.split()
            >>> X.shape, y.shape
            ((45730, 9), (45730,))

        Perform a train/test split, also outputting NumPy arrays::

            >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
            >>> X_train, X_test, y_train, y_test = train_test_split
            >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
            ((36580, 9), (36580,), (9150, 9), (9150,))

        Output the underlying Pandas DataFrame::

            >>> df = dataset.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
    '''

    _url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
            '00265/CASP.csv')

    _features = range(9)
    _targets = [9]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        # Convert the bytes into a file-like object
        csv_file = io.BytesIO(data)

        # Load in the dataframe
        df = pd.read_csv(csv_file)

        # Put the target column at the end
        df = df[[f'F{i}' for i in range(1, 10)] + ['RMSD']]

        return df
