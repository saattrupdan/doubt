from dataset import BaseDataset
import pandas as pd

class AirfoilSelfNoise(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00291/airfoil_self_noise.dat'

    raise NotImplementedError

class BikeSharing(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00275/Bike-Sharing-Dataset.zip'

    raise NotImplementedError

class BlogFeedback(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/BlogFeedback '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00304/BlogFeedback.zip'

    raise NotImplementedError

class CarbonNanotubes(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Carbon+Nanotubes '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00448/carbon_nanotubes.csv'

    raise NotImplementedError

class ConcreteCompressive(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'concrete/compressive/Concrete_Data.xls'

    raise NotImplementedError

class CPUPerformance(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Computer+Hardware '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'cpu-performance/machine.data'

    raise NotImplementedError

class CyclePowerPlant(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00294/CCPP.zip'

    raise NotImplementedError

class FacebookComments(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00363/Dataset.zip'

    raise NotImplementedError

class FacebookMetrics(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Facebook+metrics '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00368/Facebook_metrics.zip'

    raise NotImplementedError

class FishBioconcentration(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/QSAR+fish+bioconcentration+factor+%28BCF%29 '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00511/QSAR_fish_BCF.zip'

    raise NotImplementedError

class FishToxicity(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/QSAR+fish+toxicity '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00504/qsar_fish_toxicity.csv'

    raise NotImplementedError

class ForestFires(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Forest+Fires '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'forest-fires/forestfires.csv'

    raise NotImplementedError

class GasTurbine(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00316/UCI%20CBM%20Dataset.zip'

    raise NotImplementedError

class NewTaipeiHousing(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00477/Real%20estate%20valuation%20data%20set.xlsx'

    raise NotImplementedError

class Parkinsons(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'parkinsons/telemonitoring/parkinsons_updrs.data'

    raise NotImplementedError

class Protein(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00265/CASP.csv'

    raise NotImplementedError

class Servo(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Servo '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'servo/servo.data'

    raise NotImplementedError

class SolarFlare(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Solar+Flare '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'solar-flare/flare.data2'

    raise NotImplementedError

class SpaceShuttle(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Challenger+USA+Space+Shuttle+O-Ring '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          'space-shuttle/o-ring-erosion-only.data'

    raise NotImplementedError

class StockPortfolio(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Stock+portfolio+performance '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00390/stock%20portfolio%20performance%20data%20set.xlsx'

    raise NotImplementedError

class Superconduct(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00464/superconduct.zip'

    raise NotImplementedError

class TehranHousing(BaseDataset):
    ''' 
    Data set includes construction cost, sale prices, project variables, and 
    economic variables corresponding to real estate single-family residential 
    apartments in Tehran, Iran.
    
    Source:
        https://archive.ics.uci.edu/ml/datasets/Residential+Building+Data+Set 
    '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00437/Residential-Building-Data-Set.xlsx'

    def prep_data(self, data: bytes) -> pd.DataFrame:
        df = pd.read_excel(data, dtype = float, header = [0, 1])
        return df

class Yacht(BaseDataset):
    ''' https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics '''

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'\
          '00243/yacht_hydrodynamics.data'

    raise NotImplementedError
