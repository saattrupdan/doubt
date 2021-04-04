'''Nanotube data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import io


class Nanotube(BaseDataset):
    __doc__ = f'''
    CASTEP can simulate a wide range of properties of materials proprieties
    using density functional theory (DFT). DFT is the most successful method
    calculates atomic coordinates faster than other mathematical approaches,
    and it also reaches more accurate results. The dataset is generated with
    CASTEP using CNT geometry optimization. Many CNTs are simulated in CASTEP,
    then geometry optimizations are calculated. Initial coordinates of all
    carbon atoms are generated randomly. Different chiral vectors are used for
    each CNT simulation.

    The atom type is selected as carbon, bond length is used as 1.42 AÂ°
    (default value). CNT calculation parameters are used as default
    parameters. To finalize the computation, CASTEP uses a parameter named
    as elec_energy_tol (electrical energy tolerance) (default 1x10-5 eV)
    which represents that the change in the total energy from one iteration to
    the next remains below some tolerance value per atom for a few
    self-consistent field steps. Initial atomic coordinates (u, v, w), chiral
    vector (n, m) and calculated atomic coordinates (u, v, w) are
    obtained from the output files.

    {BASE_DATASET_DESCRIPTION}

    Features:
        Chiral indice n (int):
            n parameter of the selected chiral vector
        Chiral indice m (int):
            m parameter of the selected chiral vector
        Initial atomic coordinate u (float):
            Randomly generated u parameter of the initial atomic coordinates
            of all carbon atoms.
        Initial atomic coordinate v (float):
            Randomly generated v parameter of the initial atomic coordinates
            of all carbon atoms.
        Initial atomic coordinate w (float):
            Randomly generated w parameter of the initial atomic coordinates
            of all carbon atoms.

    Targets:
        Calculated atomic coordinates u (float):
           Calculated u parameter of the atomic coordinates of all
           carbon atoms
        Calculated atomic coordinates v (float):
           Calculated v parameter of the atomic coordinates of all
           carbon atoms
        Calculated atomic coordinates w (float):
           Calculated w parameter of the atomic coordinates of all
           carbon atoms

    Sources:
        https://archive.ics.uci.edu/ml/datasets/Carbon+Nanotubes
        https://doi.org/10.1007/s00339-016-0153-1
        https://doi.org/10.17341/gazimmfd.337642

    Examples:
        Load in the data set:
        >>> dataset = Nanotube()
        >>> dataset.shape
        (10721, 8)

        Split the data set into features and targets, as NumPy arrays:
        >>> X, y = dataset.split()
        >>> X.shape, y.shape
        ((10721, 5), (10721, 3))

        Perform a train/test split, also outputting NumPy arrays:
        >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
        >>> X_train, X_test, y_train, y_test = train_test_split
        >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
        ((8542, 5), (8542, 3), (2179, 5), (2179, 3))

        Output the underlying Pandas DataFrame:
        >>> df = dataset.to_pandas()
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>

        Remember to close the dataset again after use, to close the cache:
        >>> dataset.close()
    '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00448/carbon_nanotubes.csv'

    features = range(5)
    targets = [5, 6, 7]

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
        df = pd.read_csv(csv_file, sep=';', decimal=',')
        return df
