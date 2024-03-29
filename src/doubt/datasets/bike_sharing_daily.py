"""Daily bike sharing data set.

This data set is from the UCI data set archive, with the description being the original
description verbatim. Some feature names may have been altered, based on the
description.
"""

import io
import zipfile

import pandas as pd

from .dataset import BASE_DATASET_DESCRIPTION, BaseDataset


class BikeSharingDaily(BaseDataset):
    __doc__ = f"""
    Bike sharing systems are new generation of traditional bike rentals where whole
    process from membership, rental and return back has become automatic. Through these
    systems, user is able to easily rent a bike from a particular position and return
    back at another position. Currently, there are about over 500 bike-sharing programs
    around the world which is composed of over 500 thousands bicycles. Today, there
    exists great interest in these systems due to their important role in traffic,
    environmental and health issues.

    Apart from interesting real world applications of bike sharing systems, the
    characteristics of data being generated by these systems make them attractive for
    the research. Opposed to other transport services such as bus or subway, the
    duration of travel, departure and arrival position is explicitly recorded in these
    systems. This feature turns bike sharing system into a virtual sensor network that
    can be used for sensing mobility in the city. Hence, it is expected that most of
    important events in the city could be detected via monitoring these data.

    {BASE_DATASET_DESCRIPTION}

    Features:
        instant (int):
            Record index
        season (int):
            The season, with 1 = winter, 2 = spring, 3 = summer and 4 = autumn
        yr (int):
            The year, with 0 = 2011 and 1 = 2012
        mnth (int):
            The month, from 1 to 12 inclusive
        holiday (int):
            Whether day is a holiday or not, binary valued
        weekday (int):
            The day of the week, from 0 to 6 inclusive
        workingday (int):
            Working day, 1 if day is neither weekend nor holiday, otherwise 0
        weathersit (int):
            Weather, encoded as

            1. Clear, few clouds, partly cloudy
            2. Mist and cloudy, mist and broken clouds, mist and few clouds
            3. Light snow, light rain and thunderstorm and scattered clouds, light rain
            and scattered clouds
            4. Heavy rain and ice pallets and thunderstorm and mist, or snow and fog
        temp (float):
            Max-min normalised temperature in Celsius, from -8 to +39
        atemp (float):
            Max-min normalised feeling temperature in Celsius, from -16 to +50
        hum (float):
            Scaled max-min normalised humidity, from 0 to 1
        windspeed (float):
            Scaled max-min normalised wind speed, from 0 to 1

    Targets:
        casual (int):
            Count of casual users
        registered (int):
            Count of registered users
        cnt (int):
            Sum of casual and registered users

    Source:
        https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

    Examples:
        Load in the data set::

            >>> dataset = BikeSharingDaily()
            >>> dataset.shape
            (731, 15)

        Split the data set into features and targets, as NumPy arrays::

            >>> X, y = dataset.split()
            >>> X.shape, y.shape
            ((731, 12), (731, 3))

        Perform a train/test split, also outputting NumPy arrays::

            >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
            >>> X_train, X_test, y_train, y_test = train_test_split
            >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
            ((574, 12), (574, 3), (157, 12), (157, 3))

        Output the underlying Pandas DataFrame::

            >>> df = dataset.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
    """

    _url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00275/Bike-Sharing-Dataset.zip"
    )

    _features = range(12)
    _targets = [12, 13, 14]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        """Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        """
        # Convert the bytes into a file-like object
        buffer = io.BytesIO(data)

        # Unzip the file and pull out day.csv as a string
        with zipfile.ZipFile(buffer, "r") as zip_file:
            csv = zip_file.read("day.csv").decode("utf-8")

        # Convert the string into a file-like object
        csv_file = io.StringIO(csv)

        # Read the file-like object into a dataframe
        cols = [0] + list(range(2, 16))
        df = pd.read_csv(csv_file, usecols=cols)
        return df
