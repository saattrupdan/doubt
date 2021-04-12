'''Stocks data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import io


class Stocks(BaseDataset):
    __doc__ = f'''
    There are three disadvantages of weighted scoring stock selection models.
    First, they cannot identify the relations between weights of stock-picking
    concepts and performances of portfolios. Second, they cannot systematically
    discover the optimal combination for weights of concepts to optimize the
    performances. Third, they are unable to meet various investors'
    preferences.

    This study aims to more efficiently construct weighted scoring stock
    selection models to overcome these disadvantages. Since the weights of
    stock-picking concepts in a weighted scoring stock selection model can be
    regarded as components in a mixture, we used the simplex centroid mixture
    design to obtain the experimental sets of weights. These sets of weights
    are simulated with US stock market historical data to obtain their
    performances. Performance prediction models were built with the simulated
    performance data set and artificial neural networks.

    Furthermore, the optimization models to reflect investors' preferences
    were built up, and the performance prediction models were employed as the
    kernel of the optimization models so that the optimal solutions can now be
    solved with optimization techniques. The empirical values of the
    performances of the optimal weighting combinations generated by the
    optimization models showed that they can meet various investors'
    preferences and outperform those of S&P's 500 not only during the
    training period but also during the testing period.

    {BASE_DATASET_DESCRIPTION}

    Features:
        bp (float):
            Large B/P
        roe (float):
            Large ROE
        sp (float):
            Large S/P
        return_rate (float):
            Large return rate in the last quarter
        market_value (float):
            Large market value
        small_risk (float):
            Small systematic risk
        orig_annual_return (float):
            Annual return
        orig_excess_return (float):
            Excess return
        orig_risk (float):
            Systematic risk
        orig_total_risk (float):
            Total risk
        orig_abs_win_rate (float):
            Absolute win rate
        orig_rel_win_rate (float):
            Relative win rate

    Targets:
        annual_return (float):
            Annual return
        excess_return (float):
            Excess return
        risk (float):
            Systematic risk
        total_risk (float):
            Total risk
        abs_win_rate (float):
            Absolute win rate
        rel_win_rate (float):
            Relative win rate

    Source:
        https://archive.ics.uci.edu/ml/datasets/Stock+portfolio+performance

    Examples:
        Load in the data set:
        >>> dataset = Stocks()
        >>> dataset.shape
        (252, 19)

        Split the data set into features and targets, as NumPy arrays:
        >>> X, y = dataset.split()
        >>> X.shape, y.shape
        ((252, 12), (252, 6))

        Perform a train/test split, also outputting NumPy arrays:
        >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
        >>> X_train, X_test, y_train, y_test = train_test_split
        >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
        ((195, 12), (195, 6), (57, 12), (57, 6))

        Output the underlying Pandas DataFrame:
        >>> df = dataset.to_pandas()
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>
    '''

    _url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
            '00390/stock%20portfolio%20performance%20data%20set.xlsx')

    _features = range(12)
    _targets = range(12, 18)

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        # Convert the bytes into a file-like object
        xlsx_file = io.BytesIO(data)

        # Load in the dataframes
        cols = ['id', 'bp', 'roe', 'sp', 'return_rate', 'market_value',
                'small_risk', 'orig_annual_return', 'orig_excess_return',
                'orig_risk', 'orig_total_risk', 'orig_abs_win_rate',
                'orig_rel_win_rate', 'annual_return', 'excess_return',
                'risk', 'total_risk', 'abs_win_rate', 'rel_win_rate']
        sheets = ['1st period', '2nd period', '3rd period', '4th period']
        dfs = pd.read_excel(xlsx_file, sheet_name=sheets, names=cols,
                            skiprows=[0, 1], header=None)

        # Concatenate the dataframes
        df = pd.concat([dfs[sheet] for sheet in sheets], ignore_index=True)

        return df
