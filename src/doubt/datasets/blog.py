"""Blog post data set.

This data set is from the UCI data set archive, with the description being the original
description verbatim. Some feature names may have been altered, based on the
description.
"""

import io
import zipfile

import pandas as pd

from .dataset import BASE_DATASET_DESCRIPTION, BaseDataset


class Blog(BaseDataset):
    __doc__ = f"""
    This data originates from blog posts. The raw HTML-documents of the blog posts were
    crawled and processed. The prediction task associated with the data is the
    prediction of the number of comments in the upcoming 24 hours. In order to simulate
    this situation, we choose a basetime (in the past) and select the blog posts that
    were published at most 72 hours before the selected base date/time. Then, we
    calculate all the features of the selected blog posts from the information that was
    available at the basetime, therefore each instance corresponds to a blog post. The
    target is the number of comments that the blog post received in the next 24 hours
    relative to the basetime.

    In the train data, the basetimes were in the years 2010 and 2011. In the test data
    the basetimes were in February and March 2012. This simulates the real-world
    situtation in which training data from the past is available to predict events in
    the future.

    The train data was generated from different basetimes that may temporally overlap.
    Therefore, if you simply split the train into disjoint partitions, the underlying
    time intervals may overlap. Therefore, the you should use the provided, temporally
    disjoint train and test splits in order to ensure that the evaluation is fair.

    {BASE_DATASET_DESCRIPTION}

    Features:
        Features 0-49 (float):
            50 features containing the average, standard deviation, minimum, maximum
            and median of feature 50-59 for the source of the current blog post, by
            which we mean the blog on which the post appeared. For example,
            myblog.blog.org would be the source of the post
            myblog.blog.org/post_2010_09_10
        Feature 50 (int):
            Total number of comments before basetime
        Feature 51 (int):
            Number of comments in the last 24 hours before the basetime
        Feature 52 (int):
            If T1 is the datetime 48 hours before basetime and T2 is the datetime 24
            hours before basetime, then this is the number of comments in the time
            period between T1 and T2
        Feature 53 (int):
            Number of comments in the first 24 hours after the publication of the blog
            post, but before basetime
        Feature 54 (int):
            The difference between Feature 51 and Feature 52
        Features 55-59 (int):
            The same thing as Features 50-51, but for links (trackbacks) instead of
            comments
        Feature 60 (float):
            The length of time between the publication of the blog post and basetime
        Feature 61 (int):
            The length of the blog post
        Features 62-261 (int):
            The 200 bag of words features for 200 frequent words of the text of the
            blog post
        Features 262-268 (int):
            Binary indicators for the weekday (Monday-Sunday) of the basetime
        Features 269-275 (int):
            Binary indicators for the weekday (Monday-Sunday) of the date of
            publication of the blog post
        Feature 276 (int):
            Number of parent pages: we consider a blog post P as a parent of blog post
            B if B is a reply (trackback) to P
        Features 277-279 (float):
            Minimum, maximum and average of the number of comments the parents received

    Targets:
        int:
            The number of comments in the next 24 hours (relative to baseline)

    Source:
        https://archive.ics.uci.edu/ml/datasets/BlogFeedback

    Examples:
        Load in the data set::

            >>> dataset = Blog()
            >>> dataset.shape
            (52397, 281)

        Split the data set into features and targets, as NumPy arrays::

            >>> X, y = dataset.split()
            >>> X.shape, y.shape
            ((52397, 279), (52397,))

        Perform a train/test split, also outputting NumPy arrays::

            >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
            >>> X_train, X_test, y_train, y_test = train_test_split
            >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
            ((41949, 279), (41949,), (10448, 279), (10448,))

        Output the underlying Pandas DataFrame::

            >>> df = dataset.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
    """

    _url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00304/BlogFeedback.zip"
    )

    _features = range(279)
    _targets = [279]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        """Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        """
        # Convert the bytes into a file-like object
        buffer = io.BytesIO(data)

        # Unzip the file and pull out blogData_train.csv as a string
        with zipfile.ZipFile(buffer, "r") as zip_file:
            csv = zip_file.read("blogData_train.csv").decode("utf-8")

        # Convert the string into a file-like object
        csv_file = io.StringIO(csv)

        # Read the file-like object into a dataframe
        df = pd.read_csv(csv_file, header=None)
        return df
