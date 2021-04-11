'''Fish toxicity data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import io


class FishToxicity(BaseDataset):
    __doc__ = f'''
    This dataset was used to develop quantitative regression QSAR models to
    predict acute aquatic toxicity towards the fish Pimephales promelas
    (fathead minnow) on a set of 908 chemicals. LC50 data, which is the
    concentration that causes death in 50% of test fish over a test duration
    of 96 hours, was used as model response

    {BASE_DATASET_DESCRIPTION}

    Features:
        CIC0 (float):
            Information indices
        SM1_Dz(Z) (float):
            2D matrix-based descriptors
        GATS1i (float):
            2D autocorrelations
        NdsCH (int)
            Atom-type counts
        NdssC (int)
            Atom-type counts
        MLOGP (float):
            Molecular properties

    Targets:
        LC50 (float):
            A concentration that causes death in 50% of test fish over a
            test duration of 96 hours. In -log(mol/L) units.

    Source:
        https://archive.ics.uci.edu/ml/datasets/QSAR+fish+toxicity

    Examples:
        Load in the data set:
        >>> dataset = FishToxicity()
        >>> dataset.shape
        (908, 7)

        Split the data set into features and targets, as NumPy arrays:
        >>> X, y = dataset.split()
        >>> X.shape, y.shape
        ((908, 6), (908,))

        Perform a train/test split, also outputting NumPy arrays:
        >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
        >>> X_train, X_test, y_train, y_test = train_test_split
        >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
        ((701, 6), (701,), (207, 6), (207,))

        Output the underlying Pandas DataFrame:
        >>> df = dataset.to_pandas()
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>
    '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00504/qsar_fish_toxicity.csv'

    features = range(6)
    targets = [6]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        # Convert the bytes into a file-like object
        csv_file = io.BytesIO(data)

        # Read the file-like object into a dataframe
        cols = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH',
                'NdssC', 'MLOGP', 'LC50']
        df = pd.read_csv(csv_file, sep=';', header=None, names=cols)

        return df
