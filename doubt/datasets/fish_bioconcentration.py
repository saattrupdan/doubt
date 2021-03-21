'''Fish bioconcentration data set.

This data set is from the UCI data set archive, with the description being 
the original description verbatim. Some feature names may have been altered, 
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import zipfile
import io


class FishBioconcentration(BaseDataset):
    __doc__ = f'''
    This dataset contains manually-curated experimental bioconcentration 
    factor (BCF) for 1058 molecules (continuous values). Each row contains a 
    molecule, identified by a CAS number, a name (if available), and a SMILES 
    string. Additionally, the KOW (experimental or predicted) is reported. In 
    this database, you will also find Extended Connectivity Fingerprints 
    (binary vectors of 1024 bits), to be used as independent variables to 
    predict the BCF.

    {BASE_DATASET_DESCRIPTION}

    Features:
        logkow (float):
            Octanol water paritioning coefficient (experimental or predicted,
            as indicated by ``KOW type``
        kow_exp (int):
            Indicates whether ``logKOW`` is experimental or predicted, with 1
            denoting experimental and 0 denoting predicted
        smiles_[idx] for idx = 0..125 (int):
            Encoding of SMILES string to identify the 2D molecular structure.
            The encoding is as follows, where 'x' is a padding string to
            ensure that all the SMILES strings are of the same length:
                0  = 'x'
                1  = '#'
                2  = '('
                3  = ')'
                4  = '+'
                5  = '-'
                6  = '/'
                7  = '1'
                8  = '2'
                9  = '3'
                10 = '4'
                11 = '5'
                12 = '6'
                13 = '7'
                14 = '8'
                15 = '='
                16 = '@'
                17 = 'B'
                18 = 'C'
                19 = 'F'
                20 = 'H'
                21 = 'I'
                22 = 'N'
                23 = 'O'
                24 = 'P'
                25 = 'S'
                26 = '['
                27 = '\\'
                28 = ']'
                29 = 'c'
                30 = 'i'
                31 = 'l'
                32 = 'n'
                33 = 'o'
                34 = 'r'
                35 = 's'

    Targets:
        logbcf (float): 
            Experimental fish bioconcentration factor (logarithm form)
    
    Source:
        https://archive.ics.uci.edu/ml/datasets/QSAR+fish+bioconcentration+factor+%28BCF%29

    Examples:
        Load in the data set:
        >>> dataset = FishBioconcentration()
        >>> dataset.shape
        (1054, 129)

        Split the data set into features and targets, as NumPy arrays:
        >>> X, y = dataset.split()
        >>> X.shape, y.shape
        ((1054, 128), (1054, 1))

        Perform a train/test split, also outputting NumPy arrays:
        >>> train_test_split = dataset.split(test_size = 0.2, random_seed = 42)
        >>> X_train, y_train, X_test, y_test = train_test_split
        >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
        ((819, 128), (819, 1), (235, 128), (235, 1))

        Output the underlying Pandas DataFrame:
        >>> df = dataset.to_pandas()
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>

        Remember to close the dataset again after use, to close the cache:
        >>> dataset.close()
    ''' 

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00511/QSAR_fish_BCF.zip'

    feats = range(128)
    trgts = [128]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        # Convert the bytes into a file-like object
        buffer = io.BytesIO(data)

        # Unzip the file and pull out the csv file
        with zipfile.ZipFile(buffer, 'r') as zip_file:
            csv = zip_file.read('QSAR_BCF_Kow.csv')

        # Convert the string into a file-like object
        csv_file = io.BytesIO(csv)

        # Read the file-like object into a dataframe
        cols = ['cas', 'name', 'smiles', 'logkow', 'kow_exp', 'logbcf']
        df = pd.read_csv(
            csv_file, 
            names=cols, 
            header=0, 
            usecols = [col for col in cols if col not in ['cas', 'name']]
        )

        # Drop NaNs 
        df = df.dropna()

        # Encode KOW types
        kow_types = ['pred', 'exp']
        df['kow_exp'] = df.kow_exp.map(lambda txt: kow_types.index(txt))

        # Get maximum SMILE string length and pull out all the SMILE string
        # symbols, along with a '-' symbol for padding
        max_smile = max(len(smile_string) for smile_string in df.smiles)
        smile_symbols = ['x'] + sorted({symbol for smile_string in df.smiles
                                        for symbol in set(smile_string)})

        # Pad SMILE strings
        df['smiles'] = [smile_string + 'x' * (max_smile - len(smile_string))
                        for smile_string in df.smiles]

        # Encode SMILE strings
        for idx in range(max_smile):
            fn = lambda txt: smile_symbols.index(txt[idx])
            df[f'smiles_{idx}'] = df.smiles.map(fn)

        # Drop original SMILE feature
        df = df.drop(columns='smiles')

        # Put the target variable at the end
        cols = ['logkow', 'kow_exp']
        cols += [f'smiles_{idx}' for idx in range(max_smile)]
        cols += ['logbcf']
        df = df[cols]

        return df
