"""CPU data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
"""

import io

import pandas as pd

from .dataset import BASE_DATASET_DESCRIPTION, BaseDataset


class CPU(BaseDataset):
    __doc__ = f"""
    Relative CPU Performance Data, described in terms of its cycle time,
    memory size, etc.

    {BASE_DATASET_DESCRIPTION}

    Features:
        vendor_name (string):
            Name of the vendor, 30 unique values
        model_name (string):
            Name of the model
        myct (int):
            Machine cycle time in nanoseconds
        mmin (int):
            Minimum main memory in kilobytes
        mmax (int):
            Maximum main memory in kilobytes
        cach (int):
            Cache memory in kilobytes
        chmin (int):
            Minimum channels in units
        chmax (int):
            Maximum channels in units

    Targets:
        prp (int):
            Published relative performance

    Source:
        https://archive.ics.uci.edu/ml/datasets/Computer+Hardware

    Examples:
        Load in the data set::

            >>> dataset = CPU()
            >>> dataset.shape
            (209, 9)

        Split the data set into features and targets, as NumPy arrays::

            >>> X, y = dataset.split()
            >>> X.shape, y.shape
            ((209, 8), (209,))

        Perform a train/test split, also outputting NumPy arrays::

            >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
            >>> X_train, X_test, y_train, y_test = train_test_split
            >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
            ((162, 8), (162,), (47, 8), (47,))

        Output the underlying Pandas DataFrame::

            >>> df = dataset.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
    """

    _url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "cpu-performance/machine.data"
    )

    _features = range(8)
    _targets = [8]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        """Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        """

        # Convert the bytes into a file-like object
        csv_file = io.BytesIO(data)

        # Name the columns
        cols = [
            "vendor_name",
            "model_name",
            "myct",
            "mmin",
            "mmax",
            "cach",
            "chmin",
            "chmax",
            "prp",
        ]

        # Load the file-like object into a data frame
        df = pd.read_csv(csv_file, header=None, usecols=range(9), names=cols)
        return df
