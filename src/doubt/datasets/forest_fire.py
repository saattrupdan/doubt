"""Forest fire data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
"""

import io

import pandas as pd

from .dataset import BASE_DATASET_DESCRIPTION, BaseDataset


class ForestFire(BaseDataset):
    __doc__ = f"""
    This is a difficult regression task, where the aim is to predict the
    burned area of forest fires, in the northeast region of Portugal, by
    using meteorological and other data.

    {BASE_DATASET_DESCRIPTION}

    Features:
        X (float):
            The x-axis spatial coordinate within the Montesinho park map.
            Ranges from 1 to 9.
        Y (float):
            The y-axis spatial coordinate within the Montesinho park map
            Ranges from 2 to 9.
        month (int):
            Month of the year. Ranges from 0 to 11
        day (int):
            Day of the week. Ranges from 0 to 6
        FFMC (float):
            FFMC index from the FWI system. Ranges from 18.7 to 96.20
        DMC (float):
            DMC index from the FWI system. Ranges from 1.1 to 291.3
        DC (float):
            DC index from the FWI system. Ranges from 7.9 to 860.6
        ISI (float):
            ISI index from the FWI system. Ranges from 0.0 to 56.1
        temp (float):
            Temperature in Celsius degrees. Ranges from 2.2 to 33.3
        RH (float):
            Relative humidity in %. Ranges from 15.0 to 100.0
        wind (float):
            Wind speed in km/h. Ranges from 0.4 to 9.4
        rain (float):
            Outside rain in mm/m2. Ranges from 0.0 to 6.4

    Targets:
        area (float):
            The burned area of the forest (in ha). Ranges from 0.00 to 1090.84

    Notes:
        The target variable is very skewed towards 0.0, thus it may make
        sense to model with the logarithm transform.

    Source:
        https://archive.ics.uci.edu/ml/datasets/Forest+Fires

    Examples:
        Load in the data set::

            >>> dataset = ForestFire()
            >>> dataset.shape
            (517, 13)

        Split the data set into features and targets, as NumPy arrays::

            >>> X, y = dataset.split()
            >>> X.shape, y.shape
            ((517, 12), (517,))

        Perform a train/test split, also outputting NumPy arrays::

            >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
            >>> X_train, X_test, y_train, y_test = train_test_split
            >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
            ((401, 12), (401,), (116, 12), (116,))

        Output the underlying Pandas DataFrame::

            >>> df = dataset.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
    """

    _url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "forest-fires/forestfires.csv"
    )

    _features = range(12)
    _targets = [12]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        """Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        """
        # Convert the bytes into a file-like object
        csv_file = io.BytesIO(data)

        # Read the file-like object into a dataframe
        df = pd.read_csv(csv_file)

        # Encode month
        months = [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ]
        df["month"] = df.month.map(lambda string: months.index(string))

        # Encode day
        weekdays = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        df["day"] = df.day.map(lambda string: weekdays.index(string))

        return df
