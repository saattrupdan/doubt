'''Space shuttle data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import io
import re


class SpaceShuttle(BaseDataset):
    __doc__ = f'''
    The motivation for collecting this database was the explosion of the USA
    Space Shuttle Challenger on 28 January, 1986. An investigation ensued into
    the reliability of the shuttle's propulsion system. The explosion was
    eventually traced to the failure of one of the three field joints on one
    of the two solid booster rockets. Each of these six field joints includes
    two O-rings, designated as primary and secondary, which fail when
    phenomena called erosion and blowby both occur.

    The night before the launch a decision had to be made regarding launch
    safety. The discussion among engineers and managers leading to this
    decision included concern that the probability of failure of the O-rings
    depended on the temperature t at launch, which was forecase to be 31
    degrees F. There are strong engineering reasons based on the composition
    of O-rings to support the judgment that failure probability may rise
    monotonically as temperature drops. One other variable, the pressure s
    at which safety testing for field join leaks was performed, was available,
    but its relevance to the failure process was unclear.

    Draper's paper includes a menacing figure graphing the number of field
    joints experiencing stress vs. liftoff temperature for the 23 shuttle
    flights previous to the Challenger disaster. No previous liftoff
    temperature was under 53 degrees F. Although tremendous extrapolation must
    be done from the given data to assess risk at 31 degrees F, it is obvious
    even to the layman "to foresee the unacceptably high risk created by
    launching at 31 degrees F." For more information, see Draper (1993) or the
    other previous analyses.

    The task is to predict the number of O-rings that will experience thermal
    distress for a given flight when the launch temperature is below freezing.

    {BASE_DATASET_DESCRIPTION}

    Features:
        idx (int):
            Temporal order of flight
        temp (int):
            Launch temperature in Fahrenheit
        pres (int):
            Leak-check pressure in psi
        n_risky_rings (int):
            Number of O-rings at risk on a given flight

    Targets:
        n_distressed_rings (int):
            Number of O-rings experiencing thermal distress

    Source:
        https://archive.ics.uci.edu/ml/datasets/Challenger+USA+Space+Shuttle+O-Ring

    Examples:
        Load in the data set:
        >>> dataset = SpaceShuttle()
        >>> dataset.shape
        (23, 5)

        Split the data set into features and targets, as NumPy arrays:
        >>> X, y = dataset.split()
        >>> X.shape, y.shape
        ((23, 4), (23, 1))

        Perform a train/test split, also outputting NumPy arrays:
        >>> train_test_split = dataset.split(test_size = 0.2, random_seed = 42)
        >>> X_train, y_train, X_test, y_test = train_test_split
        >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
        ((16, 4), (16, 1), (7, 4), (7, 1))

        Output the underlying Pandas DataFrame:
        >>> df = dataset.to_pandas()
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>

        Remember to close the dataset again after use, to close the cache:
        >>> dataset.close()
    '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'space-shuttle/o-ring-erosion-only.data'

    feats = range(4)
    trgts = [4]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        # Collapse whitespace
        processed_data = re.sub(r' +', ' ', data.decode('utf-8'))

        # Convert the bytes into a file-like object
        csv_file = io.StringIO(processed_data)

        # Load in dataframe
        cols = ['n_risky_rings', 'n_distressed_rings', 'temp', 'pres', 'idx']
        df = pd.read_csv(csv_file, sep=' ', names=cols)

        # Reorder columns
        df = df[['idx', 'temp', 'pres', 'n_risky_rings', 'n_distressed_rings']]

        return df
