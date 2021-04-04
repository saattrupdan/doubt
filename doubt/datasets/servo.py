'''Servo data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import io


class Servo(BaseDataset):
    __doc__ = f'''
    Data was from a simulation of a servo system.

    Ross Quinlan:

    This data was given to me by Karl Ulrich at MIT in 1986. I didn't record
    his description at the time, but here's his subsequent (1992) recollection:

    "I seem to remember that the data was from a simulation of a servo system
    involving a servo amplifier, a motor, a lead screw/nut, and a sliding
    carriage of some sort. It may have been on of the translational axes of a
    robot on the 9th floor of the AI lab. In any case, the output value is
    almost certainly a rise time, or the time required for the system to
    respond to a step change in a position set point."

    (Quinlan, ML'93)

    "This is an interesting collection of data provided by Karl Ulrich. It
    covers an extremely non-linear phenomenon - predicting the rise time of a
    servomechanism in terms of two (continuous) gain settings and two
    (discrete) choices of mechanical linkages."

    {BASE_DATASET_DESCRIPTION}

    Features:
        motor (int):
            Motor, ranges from 0 to 4 inclusive
        screw (int):
            Screw, ranges from 0 to 4 inclusive
        pgain (int):
            PGain, ranges from 3 to 6 inclusive
        vgain (int):
            VGain, ranges from 1 to 5 inclusive

    Targets:
        class (float):
            Class values, ranges from 0.13 to 7.10 inclusive

    Source:
        https://archive.ics.uci.edu/ml/datasets/Servo

    Examples:
        Load in the data set:
        >>> dataset = Servo()
        >>> dataset.shape
        (167, 5)

        Split the data set into features and targets, as NumPy arrays:
        >>> X, y = dataset.split()
        >>> X.shape, y.shape
        ((167, 4), (167,))

        Perform a train/test split, also outputting NumPy arrays:
        >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
        >>> X_train, X_test, y_train, y_test = train_test_split
        >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
        ((128, 4), (128,), (39, 4), (39,))

        Output the underlying Pandas DataFrame:
        >>> df = dataset.to_pandas()
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>

        Remember to close the dataset again after use, to close the cache:
        >>> dataset.close()
    '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'servo/servo.data'

    features = range(4)
    targets = [4]

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
        cols = ['motor', 'screw', 'pgain', 'vgain', 'class']
        df = pd.read_csv(csv_file, names=cols)

        # Encode motor and screw
        codes = ['A', 'B', 'C', 'D', 'E']
        df['motor'] = df.motor.map(lambda x: codes.index(x))
        df['screw'] = df.screw.map(lambda x: codes.index(x))

        return df
