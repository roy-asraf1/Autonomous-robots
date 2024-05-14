import sys, os, csv
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import navpy
from gnssutils import EphemerisManager
from datetime import datetime
from pandas import to_datetime
import simplekml
# explanation how to run this python script in the terminal:
# python3 task_code.py
parent_directory = os.path.split(os.getcwd())[0]
ephemeris_data_directory = os.path.join(parent_directory, 'data')
sys.path.insert(0, parent_directory)

WEEKSEC = 604800
LIGHTSPEED = 2.99792458e8
    
print("Task 1: Read the data from the file")
def read_txt_file(path: str):
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0][0] == '#':
                if 'Fix' in row[0]:
                    android_fixes = [row[1:]]
                elif 'Raw' in row[0]:
                    gps_data = [row[1:]]
            else:
                if row[0] == 'Fix':
                    android_fixes.append(row[1:])
                elif row[0] == 'Raw':
                    gps_data.append(row[1:])


    print("Task 2: Convert the data to a pandas DataFrame")
    android_fixes = pd.DataFrame(android_fixes[1:], columns = android_fixes[0])
    gps_data = pd.DataFrame(gps_data[1:], columns = gps_data[0])

    print("Android head:")
    print(android_fixes.head())

    print("gps_data head:/n")
    print(gps_data.head())

    print("Task 3: Data preprocessing")
    # Format satellite IDs
    gps_data.loc[gps_data['Svid'].str.len() == 1, 'Svid'] = '0' + gps_data['Svid']
    gps_data.loc[gps_data['ConstellationType'] == '1', 'Constellation'] = 'G'
    gps_data.loc[gps_data['ConstellationType'] == '3', 'Constellation'] = 'R'
    gps_data['SvName'] = gps_data['Constellation'] + gps_data['Svid']

    print("After formatting satellite IDs: ")
    print(gps_data.head())


    # Remove all non-GPS gps_data
    gps_data = gps_data.loc[gps_data['Constellation'] == 'G']

    print("After removing non-GPS gps_data: ")
    print(gps_data.head())


    print("Task 4: Data conversion")

    # Convert columns to numeric representation
    gps_data['Cn0DbHz'] = pd.to_numeric(gps_data['Cn0DbHz'])
    gps_data['TimeNanos'] = pd.to_numeric(gps_data['TimeNanos'])
    gps_data['FullBiasNanos'] = pd.to_numeric(gps_data['FullBiasNanos'])
    gps_data['ReceivedSvTimeNanos']  = pd.to_numeric(gps_data['ReceivedSvTimeNanos'])
    gps_data['PseudorangeRateMetersPerSecond'] = pd.to_numeric(gps_data['PseudorangeRateMetersPerSecond'])
    gps_data['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(gps_data['ReceivedSvTimeUncertaintyNanos'])

    print("After converting columns to numeric representation: ")
    print(gps_data.head())


    print("Task 5: Data transformation")

    # A few measurement values are not provided by all phones
    # We'll check for them and initialize them with zeros if missing
    if 'BiasNanos' in gps_data.columns:
        gps_data['BiasNanos'] = pd.to_numeric(gps_data['BiasNanos'])
    else:
        gps_data['BiasNanos'] = 0
    if 'TimeOffsetNanos' in gps_data.columns:
        gps_data['TimeOffsetNanos'] = pd.to_numeric(gps_data['TimeOffsetNanos'])
    else:
        gps_data['TimeOffsetNanos'] = 0

    print("After checking for missing columns: ")
    print(gps_data.columns)


    print("Task 6: Data analysis")

    gps_data['GpsTimeNanos'] = gps_data['TimeNanos'] - (gps_data['FullBiasNanos'] - gps_data['BiasNanos'])
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
    gps_data['UnixTime'] = pd.to_datetime(gps_data['GpsTimeNanos'], utc = True, origin=gpsepoch)
    gps_data['UnixTime'] = gps_data['UnixTime']

    print("After calculating the Unix time of each measurement: ")
    print(gps_data.head())

    print("Task 7: Data visualization")

    # Split data into measurement epochs
    gps_data['Epoch'] = 0
    gps_data.loc[gps_data['UnixTime'] - gps_data['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    gps_data['Epoch'] = gps_data['Epoch'].cumsum()

    print("After splitting data into measurement epochs: ")
    print(gps_data.head())

    gps_data['tRxGnssNanos'] = gps_data['TimeNanos'] + gps_data['TimeOffsetNanos'] - (gps_data['FullBiasNanos'].iloc[0] + gps_data['BiasNanos'].iloc[0])
    gps_data['GpsWeekNumber'] = np.floor(1e-9 * gps_data['tRxGnssNanos'] / WEEKSEC)
    gps_data['tRxSeconds'] = 1e-9*gps_data['tRxGnssNanos'] - WEEKSEC * gps_data['GpsWeekNumber']
    gps_data['tTxSeconds'] = 1e-9*(gps_data['ReceivedSvTimeNanos'] + gps_data['TimeOffsetNanos'])
    # Calculate pseudorange in seconds
    gps_data['prSeconds'] = gps_data['tRxSeconds'] - gps_data['tTxSeconds']

    print("After calculating pseudorange in seconds: ")
    # Now we can convert to meters

    # Conver to meters
    gps_data['PrM'] = LIGHTSPEED * gps_data['prSeconds']
    gps_data['PrSigmaM'] = LIGHTSPEED * 1e-9 * gps_data['ReceivedSvTimeUncertaintyNanos']
    return gps_data, android_fixes

    
# set the panda option to display all columns
pd.set_option('display.max_columns', None)



# The function reads the data from the file and returns two pandas DataFrames: one for the GPS measurements and one for the Android fixes.
# The input is the file path to the input data and the output is two pandas DataFrames: one for the GPS measurements and one for the Android fixes.
def generate_satellite_positions(gps_data: pd.DataFrame) -> tuple:
    eph_manager = EphemerisManager(ephemeris_data_directory)


    epoch = 0
    num_sats = 0
    while num_sats < 5 :
        one_epoch = gps_data.loc[(gps_data['Epoch'] == epoch) & (gps_data['prSeconds'] < 0.1)].drop_duplicates(subset='SvName')
        timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
        one_epoch.set_index('SvName', inplace=True)
        num_sats = len(one_epoch.index)
        epoch += 1

    sats = one_epoch.index.unique().tolist()
    ephemeris = eph_manager.get_ephemeris(timestamp, sats)
    print(timestamp)
    print(one_epoch[['UnixTime', 'tTxSeconds', 'GpsWeekNumber']])

    sv_position = calculate_satellite_position(ephemeris, timestamp, one_epoch)
    return eph_manager, sv_position, one_epoch


# Calculate satellite positions using ephemeris data and pseudo-ranges
def calculate_satellite_position(ephemeris, transmit_time,one_epoch) -> pd.DataFrame: 
    mu = 3.986005e14
    OmegaDot_e = 7.2921151467e-5
    F = -4.442807633e-10
    sv_position = pd.DataFrame()
    sv_position['sv']= ephemeris.index
    sv_position.set_index('sv', inplace=True)
    sv_position['GPS time'] = transmit_time
    transmit_time = one_epoch['tTxSeconds']
    sv_position['t_k'] = transmit_time - ephemeris['t_oe']
    A = ephemeris['sqrtA'].pow(2)
    n_0 = np.sqrt(mu / A.pow(3))
    n = n_0 + ephemeris['deltaN']
    M_k = ephemeris['M_0'] + n * sv_position['t_k']
    E_k = M_k
    err = pd.Series(data=[1]*len(sv_position.index))
    i = 0
    while err.abs().min() > 1e-8 and i < 10:
        new_vals = M_k + ephemeris['e']*np.sin(E_k)
        err = new_vals - E_k
        E_k = new_vals
        i += 1
        
    sinE_k = np.sin(E_k)
    cosE_k = np.cos(E_k)
    delT_r = F * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k
    delT_oc = transmit_time - ephemeris['t_oc']
    sv_position['delT_sv'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris['SVclockDriftRate'] * delT_oc.pow(2)
    
    pr = one_epoch['PrM'] + LIGHTSPEED * sv_position['delT_sv']
    pr = pr.to_numpy()
    sv_position['Psuedo-range'] = pr
    sv_position['CN0'] = one_epoch['Cn0DbHz']

    v_k = np.arctan2(np.sqrt(1-ephemeris['e'].pow(2))*sinE_k,(cosE_k - ephemeris['e']))

    Phi_k = v_k + ephemeris['omega']

    sin2Phi_k = np.sin(2*Phi_k)
    cos2Phi_k = np.cos(2*Phi_k)

    du_k = ephemeris['C_us']*sin2Phi_k + ephemeris['C_uc']*cos2Phi_k
    dr_k = ephemeris['C_rs']*sin2Phi_k + ephemeris['C_rc']*cos2Phi_k
    di_k = ephemeris['C_is']*sin2Phi_k + ephemeris['C_ic']*cos2Phi_k

    u_k = Phi_k + du_k

    r_k = A*(1 - ephemeris['e']*np.cos(E_k)) + dr_k

    i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT']*sv_position['t_k']

    x_k_prime = r_k*np.cos(u_k)
    y_k_prime = r_k*np.sin(u_k)

    Omega_k = ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e)*sv_position['t_k'] - OmegaDot_e*ephemeris['t_oe']

    sv_position['Sat.X'] = x_k_prime*np.cos(Omega_k) - y_k_prime*np.cos(i_k)*np.sin(Omega_k)
    sv_position['Sat.Y'] = x_k_prime*np.sin(Omega_k) + y_k_prime*np.cos(i_k)*np.cos(Omega_k)
    sv_position['Sat.Z'] = y_k_prime*np.sin(i_k)
    
    return sv_position

# Calculate receiver position using satellite positions and pseudo-ranges
def calculate_xyz_from_satellites(satellite_data, initial_position=np.array([0, 0, 0]), clock_bias=0) -> np.ndarray:

    satellite_coords = satellite_data[['Sat.X', 'Sat.Y', 'Sat.Z']].to_numpy()
    pseudoranges = satellite_data['Psuedo-range']

    final_position, updated_clock_bias, residual = least_squares(satellite_coords, pseudoranges, initial_position, clock_bias)
    return final_position


# Calculate receiver position using satellite positions and pseudo-ranges 
def calculate_receiver_position(gps_data: pd.DataFrame, sv_position: pd.DataFrame, one_epoch: pd.DataFrame, eph_manager) -> np.ndarray:
    b0 = 0
    x0 = np.array([0, 0, 0])
    xs = sv_position[['Sat.X', 'Sat.Y', 'Sat.Z']].to_numpy()

    # Apply satellite clock bias to correct the measured pseudorange values
    pr = sv_position["Psuedo-range"]

    x, b, dp = least_squares(xs, pr, x0, b0)

    ecef_list = []
    for epoch in gps_data['Epoch'].unique():
        one_epoch = gps_data.loc[(gps_data['Epoch'] == epoch) & (gps_data['prSeconds'] < 0.1)] 
        one_epoch = one_epoch.drop_duplicates(subset='SvName').set_index('SvName')
        
        if len(one_epoch.index) > 4:
            timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
            sats = one_epoch.index.unique().tolist()
            ephemeris = eph_manager.get_ephemeris(timestamp, sats)
            sv_position = calculate_satellite_position(ephemeris, timestamp, one_epoch)
            x = calculate_xyz_from_satellites(sv_position, x, b)
            ecef_list.append(x)

    # Perform coordinate transformations using the Navpy library

    ecef_array = np.stack(ecef_list, axis=0)
    return ecef_array




def least_squares(xs, measured_pseudorange, x0, b0) -> tuple:
    dx = 100*np.ones(3)
    b = b0
    # set up the G matrix with the right dimensions. We will later replace the first 3 columns
    # note that b here is the clock bias in meters equivalent, so the actual clock bias is b/LIGHTSPEED
    G = np.ones((measured_pseudorange.size, 4))
    iterations = 0
    
    while np.linalg.norm(dx) > 1e-3:
        # Eq. (2):
        r = np.linalg.norm(xs - x0, axis=1)
        # Eq. (1):
        phat = r + b0
        # Eq. (3):
        deltaP = measured_pseudorange - phat
        G[:, 0:3] = -(xs - x0) / r[:, None]
        # Eq. (4):
        sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ deltaP
        # Eq. (5):
        dx = sol[0:3]
        db = sol[3]
        x0 = x0 + dx
        b0 = b0 + db
    norm_dp = np.linalg.norm(deltaP)
    return x0, b0, norm_dp

def transform_ecef_to_lla(ecef_coords: np.ndarray) -> pd.DataFrame:
    """
    Converts ECEF coordinates to LLA (Latitude, Longitude, Altitude) using coordinate transformation.

    Args:
        ecef_coords: Numpy array of ECEF coordinates.

    Returns:
        pd.DataFrame: Dataframe with columns for Latitude, Longitude, and Altitude.
    """
    # Apply transformation using a conditional expression for either single or multiple points
    lla_values = np.stack(navpy.ecef2lla(ecef_coords), axis=0).reshape(-1, 3) if ecef_coords.ndim == 1 else np.stack(navpy.ecef2lla(ecef_coords), axis=1)

    # Create a DataFrame from the transformed coordinates
    lla_frame = pd.DataFrame(lla_values, columns=['Latitude', 'Longitude', 'Altitude'])
    return lla_frame

#Converts ECEF coordinates to LLA (Latitude, Longitude, Altitude) using coordinate transformation.
#The input is a numpy array of ECEF coordinates and the output is a pandas DataFrame of LLA coordinates.


# Generate a KML file from the LLA coordinates
def create_kml_from_coordinates(coordinates_array: np.ndarray) -> simplekml.Kml:
    """
    Creates a KML file from an array of latitude, longitude, and altitude coordinates.

    Args:
        coordinates_array: Numpy array containing latitude, longitude, and altitude data.

    Returns:
        simplekml.Kml: KML object containing points for each set of coordinates.
    """
    kml_document = simplekml.Kml()
    coord_list = []

    # Initialize the index for while loop
    index = 0
    while index < len(coordinates_array):
        lat = coordinates_array[index][0]
        lon = coordinates_array[index][1]
        alt = coordinates_array[index][2]
        coord_list.append((lat, lon, alt))
        index += 1

    # Process each coordinate tuple to create KML points
    idx = 0
    while idx < len(coord_list):
        coord = coord_list[idx]
        kml_point = kml_document.newpoint(coords=[coord])
        kml_point.name = f"Point at {coord[0]}, {coord[1]}"
        idx += 1

    return kml_document

def main(input_filepath: str):
    """
    Main function to read data, process it, and generate output files.

    Args:
        input_filepath (str): File path to the input data.
    """
    base_name: str = input_filepath.split(".")[0]
    
    # Read data from file
    measurements, android_fixes = read_txt_file(input_filepath)
    
    # Generate satellite positions and calculate receiver positions
    manager, sv_positions, one_epoch = generate_satellite_positions(measurements)
    position_x_y_z = calculate_receiver_position(measurements, sv_positions, one_epoch, manager)
    pos_x_y_z = calculate_xyz_from_satellites(sv_positions)
    
    # Transform positions to LLA
    lla_df = transform_ecef_to_lla(position_x_y_z)
    transformed_positions = transform_ecef_to_lla(np.stack(pos_x_y_z, axis=0))
    
    # Update satellite positions dataframe
    sv_positions[["Pos.X", "Pos.Y", "Pos.Z"]] = pos_x_y_z
    sv_positions[["Lat", "Lon", "Alt"]] = transformed_positions.to_numpy()[0]
    sv_positions.drop(["t_k", "delT_sv"], axis=1, inplace=True)
    sv_positions = sv_positions[["GPS time", "Sat.X", "Sat.Y", "Sat.Z", "Psuedo-range", "CN0", "Pos.X", "Pos.Y", "Pos.Z", "Lat", "Lon", "Alt"]]
    
    # Create KML from coordinates
    kml = create_kml_from_coordinates(lla_df.to_numpy())
    
    # Save outputs to files
    sv_positions.to_csv(f'{base_name}-sv_positions.csv')
    lla_df.to_csv(f'{base_name}-calculated_positions.csv')
    android_fixes.to_csv(f'{base_name}-android_positions.csv')
    kml.save(f"{base_name}-KML.kml")


if __name__ == "__main__":
    main("fixed.txt")
    main("walking.txt")
    main("driving.txt")