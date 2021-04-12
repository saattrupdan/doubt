'''Parkinsons data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import io


class Parkinsons(BaseDataset):
    __doc__ = f'''
    This dataset is composed of a range of biomedical voice measurements from
    42 people with early-stage Parkinson's disease recruited to a six-month
    trial of a telemonitoring device for remote symptom progression
    monitoring. The recordings were automatically captured in the patient's
    homes.

    Columns in the table contain subject number, subject age, subject gender,
    time interval from baseline recruitment date, motor UPDRS, total UPDRS,
    and 16 biomedical voice measures. Each row corresponds to one of 5,875
    voice recording from these individuals. The main aim of the data is to
    predict the motor and total UPDRS scores ('motor_UPDRS' and 'total_UPDRS')
    from the 16 voice measures.

    {BASE_DATASET_DESCRIPTION}

    Features:
        subject# (int):
            Integer that uniquely identifies each subject
        age (int):
            Subject age
        sex (int):
            Binary feature. Subject sex, with 0 being male and 1 female
        test_time (float):
            Time since recruitment into the trial. The integer part is the
            number of days since recruitment
        Jitter(%) (float):
            Measure of variation in fundamental frequency
        Jitter(Abs) (float):
            Measure of variation in fundamental frequency
        Jitter:RAP (float):
            Measure of variation in fundamental frequency
        Jitter:PPQ5 (float):
            Measure of variation in fundamental frequency
        Jitter:DDP (float):
            Measure of variation in fundamental frequency
        Shimmer (float):
            Measure of variation in amplitude
        Shimmer(dB) (float):
            Measure of variation in amplitude
        Shimmer:APQ3 (float):
            Measure of variation in amplitude
        Shimmer:APQ5 (float):
            Measure of variation in amplitude
        Shimmer:APQ11 (float):
            Measure of variation in amplitude
        Shimmer:DDA (float):
            Measure of variation in amplitude
        NHR (float):
            Measure of ratio of noise to tonal components in the voice
        HNR (float):
            Measure of ratio of noise to tonal components in the voice
        RPDE (float):
            A nonlinear dynamical complexity measure
        DFA (float):
            Signal fractal scaling exponent
        PPE (float):
            A nonlinear measure of fundamental frequency variation

    Targets:
        motor_UPDRS (float):
            Clinician's motor UPDRS score, linearly interpolated
        total_UPDRS (float):
            Clinician's total UPDRS score, linearly interpolated

    Source:
        https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring

    Examples:
        Load in the data set:
        >>> dataset = Parkinsons()
        >>> dataset.shape
        (5875, 22)

        Split the data set into features and targets, as NumPy arrays:
        >>> X, y = dataset.split()
        >>> X.shape, y.shape
        ((5875, 20), (5875, 2))

        Perform a train/test split, also outputting NumPy arrays:
        >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
        >>> X_train, X_test, y_train, y_test = train_test_split
        >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
        ((4668, 20), (4668, 2), (1207, 20), (1207, 2))

        Output the underlying Pandas DataFrame:
        >>> df = dataset.to_pandas()
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>
    '''

    _url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
            'parkinsons/telemonitoring/parkinsons_updrs.data')

    _features = range(20)
    _targets = [20, 21]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        # Convert the bytes into a file-like object
        csv_file = io.BytesIO(data)

        # Load in dataframe
        df = pd.read_csv(csv_file, header=0)

        # Put target columns at the end
        cols = [col for col in df.columns if col[-5:] != 'UPDRS']
        df = df[cols + ['motor_UPDRS', 'total_UPDRS']]

        return df
