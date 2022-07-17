"""Yacht data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
"""

import io

import pandas as pd

from ._dataset import BASE_DATASET_DESCRIPTION, BaseDataset


class Yacht(BaseDataset):
    __doc__ = f"""
    Prediction of residuary resistance of sailing yachts at the initial design
    stage is of a great value for evaluating the ship's performance and for
    estimating the required propulsive power. Essential inputs include the
    basic hull dimensions and the boat velocity.

    The Delft data set comprises 308 full-scale experiments, which were
    performed at the Delft Ship Hydromechanics Laboratory for that purpose.

    These experiments include 22 different hull forms, derived from a parent
    form closely related to the "Standfast 43" designed by Frans Maas.

    {BASE_DATASET_DESCRIPTION}

    Features:
        pos (float):
            Longitudinal position of the center of buoyancy, adimensional
        prismatic (float):
            Prismatic coefficient, adimensional
        displacement (float):
            Length-displacement ratio, adimensional
        beam_draught (float):
            Beam-draught ratio, adimensional
        length_beam (float):
            Length-beam ratio, adimensional
        froude_no (float):
            Froude number, adimensional

    Targets:
        resistance (float):
            Residuary resistance per unit weight of displacement, adimensional

    Source:
        https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics

    Examples:
        Load in the data set::

            >>> dataset = Yacht()
            >>> dataset.shape
            (308, 7)

        Split the data set into features and targets, as NumPy arrays::

            >>> X, y = dataset.split()
            >>> X.shape, y.shape
            ((308, 6), (308,))

        Perform a train/test split, also outputting NumPy arrays::

            >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
            >>> X_train, X_test, y_train, y_test = train_test_split
            >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
            ((235, 6), (235,), (73, 6), (73,))

        Output the underlying Pandas DataFrame::

            >>> df = dataset.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
    """

    _url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00243/yacht_hydrodynamics.data"
    )

    _features = range(6)
    _targets = [6]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        """Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        """
        # Convert the bytes into a file-like object
        txt_file = io.BytesIO(data)

        # Load it into dataframe
        cols = [
            "pos",
            "prismatic",
            "displacement",
            "beam_draught",
            "length_beam",
            "froude_no",
            "resistance",
        ]
        df = pd.read_csv(txt_file, header=None, sep=" ", names=cols)
        return df
