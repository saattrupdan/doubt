'''Gas turbine data set.

This data set is from the UCI data set archive, with the description being
the original description verbatim. Some feature names may have been altered,
based on the description.
'''

from ._dataset import BaseDataset, BASE_DATASET_DESCRIPTION

import pandas as pd
import zipfile
import io


class GasTurbine(BaseDataset):
    __doc__ = f'''
    Data have been generated from a sophisticated simulator of a Gas Turbines
    (GT), mounted on a Frigate characterized by a COmbined Diesel eLectric
    And Gas (CODLAG) propulsion plant type.

    The experiments have been carried out by means of a numerical simulator of
    a naval vessel (Frigate) characterized by a Gas Turbine (GT) propulsion
    plant. The different blocks forming the complete simulator (Propeller,
    Hull, GT, Gear Box and Controller) have been developed and fine tuned over
    the year on several similar real propulsion plants. In view of these
    observations the available data are in agreement with a possible real
    vessel.

    In this release of the simulator it is also possible to take into account
    the performance decay over time of the GT components such as GT compressor
    and turbines.

    The propulsion system behaviour has been described with this parameters:

        - Ship speed (linear function of the lever position lp).
        - Compressor degradation coefficient kMc.
        - Turbine degradation coefficient kMt.

    so that each possible degradation state can be described by a combination
    of this triple (lp,kMt,kMc).

    The range of decay of compressor and turbine has been sampled with an
    uniform grid of precision 0.001 so to have a good granularity of
    representation.

    In particular for the compressor decay state discretization the kMc
    coefficient has been investigated in the domain [1; 0.95], and the turbine
    coefficient in the domain [1; 0.975].

    Ship speed has been investigated sampling the range of feasible speed from
    3 knots to 27 knots with a granularity of representation equal to tree
    knots.

    A series of measures (16 features) which indirectly represents of the
    state of the system subject to performance decay has been acquired and
    stored in the dataset over the parameter's space.

    {BASE_DATASET_DESCRIPTION}

    Features:
        lever_position (float)
            The position of the lever
        ship_speed (float):
            The ship speed, in knots
        shaft_torque (float):
            The shaft torque of the gas turbine, in kN m
        turbine_revolution_rate (float):
            The gas turbine rate of revolutions, in rpm
        generator_revolution_rate (float):
            The gas generator rate of revolutions, in rpm
        starboard_propeller_torque (float):
            The torque of the starboard propeller, in kN
        port_propeller_torque (float):
            The torque of the port propeller, in kN
        turbine_exit_temp (float):
            Height pressure turbine exit temperature, in celcius
        inlet_temp (float):
            Gas turbine compressor inlet air temperature, in celcius
        outlet_temp (float):
            Gas turbine compressor outlet air temperature, in celcius
        turbine_exit_pres (float):
            Height pressure turbine exit pressure, in bar
        inlet_pres (float):
            Gas turbine compressor inlet air pressure, in bar
        outlet_pres (float):
            Gas turbine compressor outlet air pressure, in bar
        exhaust_pres (float):
            Gas turbine exhaust gas pressure, in bar
        turbine_injection_control (float):
            Turbine injection control, in percent
        fuel_flow (float):
            Fuel flow, in kg/s

    Targets:
        compressor_decay (type):
            Gas turbine compressor decay state coefficient
        turbine_decay (type):
            Gas turbine decay state coefficient

    Source:
        https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants

    Examples:
        Load in the data set::

            >>> dataset = GasTurbine()
            >>> dataset.shape
            (11934, 18)

        Split the data set into features and targets, as NumPy arrays::

            >>> X, y = dataset.split()
            >>> X.shape, y.shape
            ((11934, 16), (11934, 2))

        Perform a train/test split, also outputting NumPy arrays::

            >>> train_test_split = dataset.split(test_size=0.2, random_seed=42)
            >>> X_train, X_test, y_train, y_test = train_test_split
            >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
            ((9520, 16), (9520, 2), (2414, 16), (2414, 2))

        Output the underlying Pandas DataFrame::

            >>> df = dataset.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
    '''

    _url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
            '00316/UCI%20CBM%20Dataset.zip')

    _features = range(16)
    _targets = [16, 17]

    def _prep_data(self, data: bytes) -> pd.DataFrame:
        ''' Prepare the data set.

        Args:
            data (bytes): The raw data

        Returns:
            Pandas dataframe: The prepared data
        '''
        # Convert the bytes into a file-like object
        buffer = io.BytesIO(data)

        # Unzip the file and pull out the txt file
        with zipfile.ZipFile(buffer, 'r') as zip_file:
            txt = zip_file.read('UCI CBM Dataset/data.txt')

        # Decode text and replace initial space on each line
        txt = txt[3:].decode('utf-8').replace('\n   ', '\n')

        # Convert the remaining triple spaces into commas, to make loading
        # it as a csv file easier
        txt = txt.replace('   ', ',')

        # Convert the string into a file-like object
        csv_file = io.StringIO(txt)

        # Read the file-like object into a dataframe
        cols = ['lever_position', 'ship_speed', 'shaft_torque',
                'turbine_revolution_rate', 'generator_revolution_rate',
                'starboard_propeller_torque', 'port_propeller_torque',
                'turbine_exit_temp', 'inlet_temp', 'outlet_temp',
                'turbine_exit_pres', 'inlet_pres', 'outlet_pres',
                'exhaust_pres', 'turbine_injection_control', 'fuel_flow',
                'compressor_decay', 'turbine_decay']
        df = pd.read_csv(csv_file, header=None, names=cols)

        return df
