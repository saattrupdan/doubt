"""Concrete data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
"""

import io

import pandas as pd

from ._dataset import BASE_DATASET_DESCRIPTION, BaseDataset


class Concrete(BaseDataset):
    __doc__ = f"""
    Concrete is the most important material in civil engineering. The concrete
    compressive strength is a highly nonlinear function of age and
    ingredients.

    {BASE_DATASET_DESCRIPTION}

    Features:
        Cement (float):
            Kg of cement in an m3 mixture
        Blast Furnace Slag (float):
            Kg of blast furnace slag in an m3 mixture
        Fly Ash (float):
            Kg of fly ash in an m3 mixture
        Water (float):
            Kg of water in an m3 mixture
        Superplasticiser (float):
            Kg of superplasticiser in an m3 mixture
        Coarse Aggregate (float):
            Kg of coarse aggregate in an m3 mixture
        Fine Aggregate (float):
            Kg of fine aggregate in an m3 mixture
        Age (int):
            Age in days, between 1 and 365 inclusive

    Targets:
        Concrete Compressive Strength (float):
            Concrete compressive strength in megapascals

    Source:
        https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength

    Examples:
        Load in the data set::

            >>> dataset = Concrete()
            >>> dataset.shape
            (1030, 9)

        Split the data set into features and targets, as NumPy arrays::

            >>> X, y = dataset.split()
            >>> X.shape, y.shape
            ((1030, 8), (1030,))

        Perform a train/test split, also outputting NumPy arrays::

            >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
            >>> X_train, X_test, y_train, y_test = train_test_split
            >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
            ((807, 8), (807,), (223, 8), (223,))

        Output the underlying Pandas DataFrame::

            >>> df = dataset.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
    """

    _url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "concrete/compressive/Concrete_Data.xls"
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
        xls_file = io.BytesIO(data)

        # Load the file-like object into a data frame
        df = pd.read_excel(xls_file)
        return df
