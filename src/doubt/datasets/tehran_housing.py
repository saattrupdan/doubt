"""Tehran housing data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
"""

import io

import pandas as pd

from ._dataset import BASE_DATASET_DESCRIPTION, BaseDataset


class TehranHousing(BaseDataset):
    __doc__ = f"""
    Data set includes construction cost, sale prices, project variables, and
    economic variables corresponding to real estate single-family residential
    apartments in Tehran, Iran.

    {BASE_DATASET_DESCRIPTION}

    Features:
        start_year (int):
            Start year in the Persian calendar
        start_quarter (int)
            Start quarter in the Persian calendar
        completion_year (int)
            Completion year in the Persian calendar
        completion_quarter (int)
            Completion quarter in the Persian calendar
        V-1..V-8 (floats):
            Project physical and financial variables
        V-11-1..29-1 (floats):
            Economic variables and indices in time, lag 1
        V-11-2..29-2 (floats):
            Economic variables and indices in time, lag 2
        V-11-3..29-3 (floats):
            Economic variables and indices in time, lag 3
        V-11-4..29-4 (floats):
            Economic variables and indices in time, lag 4
        V-11-5..29-5 (floats):
            Economic variables and indices in time, lag 5

    Targets:
        construction_cost (float)
        sale_price (float)

    Source:
        https://archive.ics.uci.edu/ml/datasets/Residential+Building+Data+Set

    Examples:
        Load in the data set::

            >>> dataset = TehranHousing()
            >>> dataset.shape
            (371, 109)

        Split the data set into features and targets, as NumPy arrays::

            >>> X, y = dataset.split()
            >>> X.shape, y.shape
            ((371, 107), (371, 2))

        Perform a train/test split, also outputting NumPy arrays::

            >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
            >>> X_train, X_test, y_train, y_test = train_test_split
            >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
            ((288, 107), (288, 2), (83, 107), (83, 2))

        Output the underlying Pandas DataFrame::

            >>> df = dataset.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
    """

    _url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00437/Residential-Building-Data-Set.xlsx"
    )

    _features = range(107)
    _targets = [107, 108]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        """Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        """
        # Convert the bytes into a file-like object
        xlsx_file = io.BytesIO(data)

        # Load it into dataframe
        cols = (
            ["start_year", "start_quarter", "completion_year", "completion_quarter"]
            + [f"V-{i}" for i in range(1, 9)]
            + [f"V-{i}-{j}" for j in range(1, 6) for i in range(11, 30)]
            + ["construction_cost", "sale_price"]
        )
        df = pd.read_excel(xlsx_file, skiprows=[0, 1], names=cols)
        return df
