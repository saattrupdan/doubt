"""Facebook metrics data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
"""

import io
import zipfile

import pandas as pd

from ._dataset import BASE_DATASET_DESCRIPTION, BaseDataset


class FacebookMetrics(BaseDataset):
    __doc__ = f"""
    The data is related to posts' published during the year of 2014 on the
    Facebook's page of a renowned cosmetics brand.

    {BASE_DATASET_DESCRIPTION}

    Features:
        page_likes(int):
            The total number of likes of the Facebook page at the given time.
        post_type (int):
            The type of post. Here 0 means 'Photo', 1 means 'Status', 2 means
            'Link' and 3 means 'Video'
        post_category (int):
            The category of the post.
        post_month (int):
            The month the post was posted, from 1 to 12 inclusive.
        post_weekday (int):
            The day of the week the post was posted, from 1 to 7 inclusive.
        post_hour (int):
            The hour the post was posted, from 0 to 23 inclusive
        paid (int):
            Binary feature, whether the post was paid for.

    Targets:
        total_reach (int):
            The lifetime post total reach.
        total_impressions (int):
            The lifetime post total impressions.
        engaged_users (int):
            The lifetime engaged users.
        post_consumers (int):
            The lifetime post consumers.
        post_consumptions (int):
            The lifetime post consumptions.
        post_impressions (int):
            The lifetime post impressions by people who liked the page.
        post_reach (int):
            The lifetime post reach by people who liked the page.
        post_engagements (int):
            The lifetime people who have liked the page and engaged with
            the post.
        comments (int):
            The number of comments.
        shares (int):
            The number of shares.
        total_interactions (int):
            The total number of interactions

    Source:
        https://archive.ics.uci.edu/ml/datasets/Facebook+metrics

    Examples:
        Load in the data set::

            >>> dataset = FacebookMetrics()
            >>> dataset.shape
            (500, 18)

        Split the data set into features and targets, as NumPy arrays::

            >>> X, y = dataset.split()
            >>> X.shape, y.shape
            ((500, 7), (500, 11))

        Perform a train/test split, also outputting NumPy arrays::

            >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
            >>> X_train, X_test, y_train, y_test = train_test_split
            >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
            ((388, 7), (388, 11), (112, 7), (112, 11))

        Output the underlying Pandas DataFrame::

            >>> df = dataset.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
    """

    _url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00368/Facebook_metrics.zip"
    )

    _features = range(7)
    _targets = range(7, 18)

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        """Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        """
        # Convert the bytes into a file-like object
        buffer = io.BytesIO(data)

        # Unzip the file and pull out the csv file
        with zipfile.ZipFile(buffer, "r") as zip_file:
            csv = zip_file.read("dataset_Facebook.csv")

        # Convert the bytes into a file-like object
        csv_file = io.BytesIO(csv)

        # Read the file-like object into a dataframe
        cols = [
            "page_likes",
            "post_type",
            "post_category",
            "post_month",
            "post_weekday",
            "post_hour",
            "paid",
            "total_reach",
            "total_impressions",
            "engaged_users",
            "post_consumers",
            "post_consumptions",
            "post_impressions",
            "post_reach",
            "post_engagements",
            "comments",
            "shares",
            "total_interactions",
        ]
        df = pd.read_csv(csv_file, sep=";", names=cols, header=0, index_col=False)

        # Numericalise post type
        post_types = list(df.post_type.unique())
        df["post_type"] = df.post_type.map(lambda txt: post_types.index(txt))

        return df
