import unittest
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import os
import sys
import navpy

from task_code import (
    calculate_satellite_position,
    least_squares,
    parse_and_format_file,
    calculate_ecef_coordinates,
    WEEKSEC,
    LIGHTSPEED,
    EphemerisManager
)

# Setup directory paths for data and scripts if the code not running replace your path
parent_directory = os.path.split(os.getcwd())[0]
ephemeris_data_directory = os.path.join(parent_directory, 'data')
sys.path.insert(0, parent_directory)
file_path = "driving.txt"
    

class TestGnssFunctions(unittest.TestCase):
    
    # Validate file path is valid (.txt) and as the correct format
    def test_file_path(self):
        print("\nPart 1 - Running test_file_path")
        self.assertEqual(file_path[-4:], ".txt")
        # check this file is exist
        self.assertTrue(os.path.exists(file_path))
        print("test_file_path passed successfully.")


    # Validate the file is parsed correctly
    def test_parse_and_format_file(self):        
        print("\nPart 2 - Running test_parse_and_format_file")
        data_android, data_measurements = parse_and_format_file(file_path)
        self.assertIsInstance(data_android, pd.DataFrame)
        self.assertIsInstance(data_measurements, pd.DataFrame)
        print("test_parse_and_format_file passed successfully.")

    # Check existence of all used columns that should be in the data
    def test_columns(self):
        print("\nPart 3 - Running test_columns")
        data_android, data_measurements = parse_and_format_file(file_path)
        self.assertTrue("Svid" in data_measurements.columns)
        self.assertTrue("ConstellationType" in data_measurements.columns)
        self.assertTrue("Constellation" in data_measurements.columns)
        self.assertTrue("SvName" in data_measurements.columns)
        self.assertTrue("Cn0DbHz" in data_measurements.columns)
        self.assertTrue("ConstellationType" in data_measurements.columns)
        self.assertTrue("Svid" in data_measurements.columns)
        self.assertTrue("TimeNanos" in data_measurements.columns)
        self.assertTrue("FullBiasNanos" in data_measurements.columns)
        self.assertTrue("ReceivedSvTimeNanos" in data_measurements.columns)
        self.assertTrue("PseudorangeRateMetersPerSecond" in data_measurements.columns)
        self.assertTrue("ReceivedSvTimeUncertaintyNanos" in data_measurements.columns)
        self.assertTrue("BiasNanos" in data_measurements.columns)
        self.assertTrue("TimeOffsetNanos" in data_measurements.columns)
        self.assertTrue("GpsTimeNanos" in data_measurements.columns)
        self.assertTrue("TimeNanos" in data_measurements.columns)
        self.assertTrue("tRxGnssNanos" in data_measurements.columns)
        self.assertTrue("GpsWeekNumber" in data_measurements.columns)
        self.assertTrue("ReceivedSvTimeNanos" in data_measurements.columns)
        self.assertTrue("tRxSeconds" in data_measurements.columns)
        self.assertTrue("tTxSeconds" in data_measurements.columns)
        self.assertTrue("prSeconds" in data_measurements.columns)
        self.assertTrue("PrM" in data_measurements.columns)
        self.assertTrue("PrSigmaM" in data_measurements.columns)
        print("test_columns passed successfully.")

    # Check the correctness of the satellite position calculation that have all these columns: SatPRN,GPS_time,delT_sv,Sat_X,Sat_y,Sat_z,CN0
    def test_calculate_satellite_position(self):
        print("\nPart 4 - Running test_calculate_satellite_position")
        data_android, data_measurements = parse_and_format_file(file_path)
        # Calculate satellite positions
        manager = EphemerisManager(ephemeris_data_directory)
        epoch = 0
        num_sats = 0

        while num_sats < 5: # Find epoch with at least 5 satellites
            one_epoch = data_measurements.loc[(data_measurements['Epoch'] == epoch) & (data_measurements['prSeconds'] < 0.1)]
            one_epoch = one_epoch.drop_duplicates(subset='SvName')
            timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
            one_epoch.set_index('SvName', inplace=True)
            num_sats = len(one_epoch.index)
            epoch += 1

        sats = one_epoch.index.unique().tolist()
        ephemeris = manager.get_ephemeris(timestamp, sats)

        data_measurements = calculate_satellite_position(ephemeris, one_epoch)
        self.assertTrue("GPS_time" in data_measurements.columns)
        self.assertTrue("delT_sv" in data_measurements.columns)
        self.assertTrue("Sat_X" in data_measurements.columns)
        self.assertTrue("Sat_Y" in data_measurements.columns)
        self.assertTrue("Sat_Z" in data_measurements.columns)
        self.assertTrue("CN0" in data_measurements.columns)
        print("test_calculate_satellite_position passed successfully.")



if __name__ == "__main__":
    unittest.main()
