'''Superconductivity data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import zipfile
import io


class Superconductivity(BaseDataset):
    __doc__ = f'''
    This dataset contains data on 21,263 superconductors and their relevant
    features. The goal here is to predict the critical temperature based on
    the features extracted.

    {BASE_DATASET_DESCRIPTION}

    Features:
        number_of_elements (int)
        mean_atomic_mass (float)
        wtd_mean_atomic_mass (float)
        gmean_atomic_mass (float)
        wtd_gmean_atomic_mass (float)
        entropy_atomic_mass (float)
        wtd_entropy_atomic_mass (float)
        range_atomic_mass (float)
        wtd_range_atomic_mass (float)
        std_atomic_mass (float)
        wtd_std_atomic_mass (float)
        mean_fie (float)
        wtd_mean_fie (float)
        gmean_fie (float)
        wtd_gmean_fie (float)
        entropy_fie (float)
        wtd_entropy_fie (float)
        range_fie (float)
        wtd_range_fie (float)
        std_fie (float)
        wtd_std_fie (float)
        mean_atomic_radius (float)
        wtd_mean_atomic_radius (float)
        gmean_atomic_radius (float)
        wtd_gmean_atomic_radius (float)
        entropy_atomic_radius (float)
        wtd_entropy_atomic_radius (float)
        range_atomic_radius (float)
        wtd_range_atomic_radius (float)
        std_atomic_radius (float)
        wtd_std_atomic_radius (float)
        mean_Density (float)
        wtd_mean_Density (float)
        gmean_Density (float)
        wtd_gmean_Density (float)
        entropy_Density (float)
        wtd_entropy_Density (float)
        range_Density (float)
        wtd_range_Density (float)
        std_Density (float)
        wtd_std_Density (float)
        mean_ElectronAffinity (float)
        wtd_mean_ElectronAffinity (float)
        gmean_ElectronAffinity (float)
        wtd_gmean_ElectronAffinity (float)
        entropy_ElectronAffinity (float)
        wtd_entropy_ElectronAffinity (float)
        range_ElectronAffinity (float)
        wtd_range_ElectronAffinity (float)
        std_ElectronAffinity (float)
        wtd_std_ElectronAffinity (float)
        mean_FusionHeat (float)
        wtd_mean_FusionHeat (float)
        gmean_FusionHeat (float)
        wtd_gmean_FusionHeat (float)
        entropy_FusionHeat (float)
        wtd_entropy_FusionHeat (float)
        range_FusionHeat (float)
        wtd_range_FusionHeat (float)
        std_FusionHeat (float)
        wtd_std_FusionHeat (float)
        mean_ThermalConductivity (float)
        wtd_mean_ThermalConductivity (float)
        gmean_ThermalConductivity (float)
        wtd_gmean_ThermalConductivity (float)
        entropy_ThermalConductivity (float)
        wtd_entropy_ThermalConductivity (float)
        range_ThermalConductivity (float)
        wtd_range_ThermalConductivity (float)
        std_ThermalConductivity (float)
        wtd_std_ThermalConductivity (float)
        mean_Valence (float)
        wtd_mean_Valence (float)
        gmean_Valence (float)
        wtd_gmean_Valence (float)
        entropy_Valence (float)
        wtd_entropy_Valence (float)
        range_Valence (float)
        wtd_range_Valence (float)
        std_Valence (float)
        wtd_std_Valence (float)

    Targets:
        critical_temp (float)

    Source:
        https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data

    Examples:
        Load in the data set:
        >>> dataset = Superconductivity()
        >>> dataset.shape
        (21263, 82)

        Split the data set into features and targets, as NumPy arrays:
        >>> X, y = dataset.split()
        >>> X.shape, y.shape
        ((21263, 81), (21263,))

        Perform a train/test split, also outputting NumPy arrays:
        >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
        >>> X_train, X_test, y_train, y_test = train_test_split
        >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
        ((17018, 81), (17018,), (4245, 81), (4245,))

        Output the underlying Pandas DataFrame:
        >>> df = dataset.to_pandas()
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>
    '''

    _url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
            '00464/superconduct.zip')

    _features = range(81)
    _targets = [81]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        # Convert the bytes into a file-like object
        buffer = io.BytesIO(data)

        # Unzip the file and pull out the text
        with zipfile.ZipFile(buffer, 'r') as zip_file:
            txt = zip_file.read('train.csv')

        # Convert text to csv file
        csv_file = io.BytesIO(txt)

        # Load the csv file into a dataframe
        df = pd.read_csv(csv_file)

        return df
