'''Power plant data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import zipfile
import io


class PowerPlant(BaseDataset):
    __doc__ = f'''
    The dataset contains 9568 data points collected from a Combined Cycle
    Power Plant over 6 years (2006-2011), when the power plant was set to
    work with full load. Features consist of hourly average ambient variables
    Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust
    Vacuum (V) to predict the net hourly electrical energy output (EP) of the
    plant.

    A combined cycle power plant (CCPP) is composed of gas turbines (GT),
    steam turbines (ST) and heat recovery steam generators. In a CCPP, the
    electricity is generated by gas and steam turbines, which are combined in
    one cycle, and is transferred from one turbine to another. While the
    Vacuum is colected from and has effect on the Steam Turbine, he other
    three of the ambient variables effect the GT performance.

    For comparability with our baseline studies, and to allow 5x2 fold
    statistical tests be carried out, we provide the data shuffled five times.
    For each shuffling 2-fold CV is carried out and the resulting 10
    measurements are used for statistical testing.

    {BASE_DATASET_DESCRIPTION}

    Features:
        AT (float):
            Hourly average temperature in Celsius, ranges from 1.81 to 37.11
        V (float):
            Hourly average exhaust vacuum in cm Hg, ranges from 25.36 to 81.56
        AP (float):
            Hourly average ambient pressure in millibar, ranges from 992.89
            to 1033.30
        RH (float):
            Hourly average relative humidity in percent, ranges from 25.56
            to 100.16

    Targets:
        PE (float):
            Net hourly electrical energy output in MW, ranges from 420.26
            to 495.76

    Source:
        https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant

    Examples:
        Load in the data set:
        >>> dataset = PowerPlant()
        >>> dataset.shape
        (9568, 5)

        Split the data set into features and targets, as NumPy arrays:
        >>> X, y = dataset.split()
        >>> X.shape, y.shape
        ((9568, 4), (9568,))

        Perform a train/test split, also outputting NumPy arrays:
        >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
        >>> X_train, X_test, y_train, y_test = train_test_split
        >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
        ((7615, 4), (7615,), (1953, 4), (1953,))

        Output the underlying Pandas DataFrame:
        >>> df = dataset.to_pandas()
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>
    '''

    _url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
            '00294/CCPP.zip')

    _features = range(4)
    _targets = [4]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''

        # Convert the bytes into a file-like object
        buffer = io.BytesIO(data)

        # Unzip the file and pull out the xlsx file
        with zipfile.ZipFile(buffer, 'r') as zip_file:
            xlsx = zip_file.read('CCPP/Folds5x2_pp.xlsx')

        # Convert the xlsx bytes into a file-like object
        xlsx_file = io.BytesIO(xlsx)

        # Read the file-like object into a dataframe
        df = pd.read_excel(xlsx_file)
        return df
