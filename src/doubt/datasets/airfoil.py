"""Airfoil data set.

This data set is from the UCI data set archive, with the description being the original
description verbatim. Some feature names may have been altered, based on the
description.
"""

import io

import pandas as pd

from .dataset import BASE_DATASET_DESCRIPTION, BaseDataset


class Airfoil(BaseDataset):
    __doc__ = f"""
    The NASA data set comprises different size NACA 0012 airfoils at various wind
    tunnel speeds and angles of attack. The span of the airfoil and the observer
    position were the same in all of the experiments.

    {BASE_DATASET_DESCRIPTION}

    Features:
        int:
            Frequency, in Hertzs
        float:
            Angle of attack, in degrees
        float:
            Chord length, in meters
        float:
            Free-stream velocity, in meters per second
        float:
            Suction side displacement thickness, in meters

    Targets:
        float:
            Scaled sound pressure level, in decibels

    Source:
        https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise

    Examples:
        Load in the data set::

            >>> dataset = Airfoil()
            >>> dataset.shape
            (1503, 6)

        Split the data set into features and targets, as NumPy arrays::

            >>> X, y = dataset.split()
            >>> X.shape, y.shape
            ((1503, 5), (1503,))

        Perform a train/test split, also outputting NumPy arrays::

            >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
            >>> X_train, X_test, y_train, y_test = train_test_split
            >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
            ((1181, 5), (1181,), (322, 5), (322,))

        Output the underlying Pandas DataFrame::

            >>> df = dataset.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
    """

    _url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00291/airfoil_self_noise.dat"
    )

    _features = range(5)
    _targets = [5]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        """Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        """
        # Convert the bytes into a file-like object
        csv_file = io.BytesIO(data)

        # Read the file-like object into a data frame
        df = pd.read_csv(csv_file, sep="\t", header=None)
        return df
