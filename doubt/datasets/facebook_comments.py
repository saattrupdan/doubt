'''Facebook comments data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import zipfile
import io


class FacebookComments(BaseDataset):
    __doc__ = f'''
    Instances in this dataset contain features extracted from Facebook posts.
    The task associated with the data is to predict how many comments the
    post will receive.

    {BASE_DATASET_DESCRIPTION}

    Features:
        page_popularity (int):
            Defines the popularity of support for the source of the document
        page_checkins (int):
            Describes how many individuals so far visited this place. This
            feature is only associated with places; e.g., some institution,
            place, theater, etc.
        page_talking_about (int):
            Defines the daily interest of individuals towards source of the
            document/post. The people who actually come back to the page,
            after liking the page. This include activities such as comments,
            likes to a post, shares etc., by visitors to the page
        page_category (int):
            Defines the category of the source of the document; e.g., place,
            institution, branch etc.
        agg[n] for n=0..24 (float):
            These features are aggreagted by page, by calculating min, max,
            average, median and standard deviation of essential features
        cc1 (int):
            The total number of comments before selected base date/time
        cc2 (int):
            The number of comments in the last 24 hours, relative to base
            date/time
        cc3 (int):
            The number of comments in the last 48 to last 24 hours relative
            to base date/time
        cc4 (int):
            The number of comments in the first 24 hours after the publication
            of post but before base date/time
        cc5 (int):
            The difference between cc2 and cc3
        base_time (int):
            Selected time in order to simulate the scenario, ranges from 0
            to 71
        post_length (int):
            Character count in the post
        post_share_count (int):
            This feature counts the number of shares of the post, how many
            people had shared this post onto their timeline
        post_promotion_status (int):
            Binary feature. To reach more people with posts in News Feed,
            individuals can promote their post and this feature indicates
            whether the post is promoted or not
        h_local (int):
            This describes the hours for which we have received the target
            variable/comments. Ranges from 0 to 23
        day_published[n] for n=0..6 (int):
            Binary feature. This represents the day (Sunday-Saturday) on
            which the post was published
        day[n] for n=0..6 (int):
            Binary feature. This represents the day (Sunday-Saturday) on
            selected base date/time

    Targets:
        ncomments (int): The number of comments in the next `h_local` hours

    Source:
        https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset

    Examples:
        Load in the data set:
        >>> dataset = FacebookComments()
        >>> dataset.shape
        (199030, 54)

        Split the data set into features and targets, as NumPy arrays:
        >>> X, y = dataset.split()
        >>> X.shape, y.shape
        ((199030, 54), (199030,))

        Perform a train/test split, also outputting NumPy arrays:
        >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
        >>> X_train, y_train, X_test, y_test = train_test_split
        >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
        ((159288, 54), (159288,), (39742, 54), (39742,))

        Output the underlying Pandas DataFrame:
        >>> df = dataset.to_pandas()
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>

        Remember to close the dataset again after use, to close the cache:
        >>> dataset.close()
    '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00363/Dataset.zip'

    feats = range(54)
    trgts = [53]

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
            csv = zip_file.read('Dataset/Training/Features_Variant_5.csv')

        # Convert the string into a file-like object
        csv_file = io.BytesIO(csv)

        # Name the columns
        cols = ['page_popularity', 'page_checkins', 'page_talking_about',
                'page_category'] + \
               [f'agg{n}' for n in range(25)] + \
               ['cc1', 'cc2', 'cc3', 'cc4', 'cc5', 'base_time', 'post_length',
                'post_share_count', 'post_promotion_status', 'h_local'] + \
               [f'day_published{n}' for n in range(7)] + \
               [f'day{n}' for n in range(7)] + \
               ['ncomments']

        # Read the file-like object into a dataframe
        df = pd.read_csv(csv_file, header=None, names=cols)
        return df
