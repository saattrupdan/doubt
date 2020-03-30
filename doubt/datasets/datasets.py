from ._dataset import BaseDataset

import pandas as pd

class AirfoilSelfNoise(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00291/airfoil_self_noise.dat'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class BikeSharing(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00275/Bike-Sharing-Dataset.zip'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class BlogFeedback(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/BlogFeedback '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00304/BlogFeedback.zip'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class CarbonNanotubes(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Carbon+Nanotubes '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00448/carbon_nanotubes.csv'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class ConcreteCompressive(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'concrete/compressive/Concrete_Data.xls'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class CPUPerformance(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Computer+Hardware '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'cpu-performance/machine.data'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class CyclePowerPlant(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00294/CCPP.zip'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class FacebookComments(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00363/Dataset.zip'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class FacebookMetrics(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Facebook+metrics '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00368/Facebook_metrics.zip'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class FishBioconcentration(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/QSAR+fish+bioconcentration+factor+%28BCF%29 '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00511/QSAR_fish_BCF.zip'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class FishToxicity(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/QSAR+fish+toxicity '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00504/qsar_fish_toxicity.csv'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class ForestFires(BaseDataset):
    ''' 
    This is a difficult regression task, where the aim is to predict the 
    burned area of forest fires, in the northeast region of Portugal, by 
    using meteorological and other data.

    Features:
        X (float): 
            The x-axis spatial coordinate within the Montesinho park map.
            Ranges from 1 to 9.
        Y (float): 
            The y-axis spatial coordinate within the Montesinho park map
            Ranges from 2 to 9.
        month (str):
            Month of the year. Ranges from 'jan' to 'dec'
        day (str):
            Day of the year. Ranges from 'mon' to 'sun'
        FFMC (float):
            FFMC index from the FWI system. Ranges from 18.7 to 96.20
        DMC (float):
            DMC index from the FWI system. Ranges from 1.1 to 291.3
        DC (float):
            DC index from the FWI system. Ranges from 7.9 to 860.6
        ISI (float):
            ISI index from the FWI system. Ranges from 0.0 to 56.1
        temp (float):
            Temperature in Celsius degrees. Ranges from 2.2 to 33.3
        RH (float):
            Relative humidity in %. Ranges from 15.0 to 100.0
        wind (float):
            Wind speed in km/h. Ranges from 0.4 to 9.4
        rain (float):
            Outside rain in mm/m2. Ranges from 0.0 to 6.4
    
    Targets:
        area (float):
            The burned area of the forest (in ha). Ranges from 0.00 to 1090.84

    Notes:
        The target variable is very skewed towards 0.0, thus it may make
        sense to model with the logarithm transform.

    Source:
        https://archive.ics.uci.edu/ml/datasets/Forest+Fires 
    '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'forest-fires/forestfires.csv'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class GasTurbine(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00316/UCI%20CBM%20Dataset.zip'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class NewTaipeiHousing(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00477/Real%20estate%20valuation%20data%20set.xlsx'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class Parkinsons(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'parkinsons/telemonitoring/parkinsons_updrs.data'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class Protein(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00265/CASP.csv'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class Servo(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Servo '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'servo/servo.data'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class SolarFlare(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Solar+Flare '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'solar-flare/flare.data2'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class SpaceShuttle(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Challenger+USA+Space+Shuttle+O-Ring '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'space-shuttle/o-ring-erosion-only.data'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class StockPortfolio(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Stock+portfolio+performance '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00390/stock%20portfolio%20performance%20data%20set.xlsx'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class Superconduct(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00464/superconduct.zip'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

class TehranHousing(BaseDataset):
    ''' 
    Data set includes construction cost, sale prices, project variables, and 
    economic variables corresponding to real estate single-family residential 
    apartments in Tehran, Iran.

    Features:
        float: 8 project physical and financial variables
        float: 19 economic variables and indices in 5 time lag numbers

    Targets:
        float: Construction cost
        float: Sale price
    
    Source:
        https://archive.ics.uci.edu/ml/datasets/Residential+Building+Data+Set 
    '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00437/Residential-Building-Data-Set.xlsx'

    feats = range(105)
    trgts = [105, 106]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        df = pd.read_excel(data, dtype = float, header = [0, 1])
        return df

class Yacht(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00243/yacht_hydrodynamics.data'

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        raise NotImplementedError

if __name__ == '__main__':
    tehran = TehranHousing()
    print(tehran.columns)
    print(tehran.head())
