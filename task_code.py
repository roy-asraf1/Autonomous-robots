import sys, os, csv
import simplekml
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import navpy
from gnssutils import EphemerisManager
import matplotlib.pyplot as plt
import argparse  # Import argparse for command-line argument parsing
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarnings

# Constants Definitions
WEEKSEC = 604800
LIGHTSPEED = 2.99792458e8


def main(file_path):
    # Setup directory paths for data and scripts if the code not running replace your path
    parent_directory = os.path.split(os.getcwd())[0]
    ephemeris_data_directory = os.path.join(parent_directory, 'data')
    sys.path.insert(0, parent_directory)

    android_fixes, measurements = parse_and_format_file(file_path)

    # Calculate satellite positions
    manager = EphemerisManager(ephemeris_data_directory)
    epoch = 0
    num_sats = 0

    while num_sats < 5: # Find epoch with at least 5 satellites
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)]
        one_epoch = one_epoch.drop_duplicates(subset='SvName')
        timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
        one_epoch.set_index('SvName', inplace=True)
        num_sats = len(one_epoch.index)
        epoch += 1

    sats = one_epoch.index.unique().tolist()
    ephemeris = manager.get_ephemeris(timestamp, sats)

    sv_position = calculate_satellite_position(ephemeris, one_epoch)
    sv_position.to_csv("satellite_positions_export.csv", sep=',')

    ecef_list = calculate_ecef_coordinates(measurements, manager, sv_position, timestamp, sats)    

    # Convert ECEF coordinates to LLA and NED
    ecef_array = np.stack(ecef_list, axis=0)
    lla_array = np.stack(navpy.ecef2lla(ecef_array), axis=1)
    ref_lla = lla_array[0, :]
    ned_array = navpy.ecef2ned(ecef_array, ref_lla[0], ref_lla[1], ref_lla[2])

    ned_df = pd.DataFrame(ned_array, columns=['N', 'E', 'D'])

    # Save LLA and NED data to CSV files
    pd.DataFrame(lla_array, columns=['Latitude', 'Longitude', 'Altitude']).to_csv('calculated_position_lla.csv', index=False)
    pd.DataFrame(ned_array, columns=['N', 'E', 'D'])

    # Create DataFrame with Pos.X, Pos.Y, Pos.Z, Lat, Lon, Alt
    final_df = pd.DataFrame({
        'Pos.X': ecef_array[:, 0],
        'Pos.Y': ecef_array[:, 1],
        'Pos.Z': ecef_array[:, 2],
        'Latitude': lla_array[:, 0],
        'Longitude': lla_array[:, 1],
        'Altitude': lla_array[:, 2]
    })

    # Export the final DataFrame to CSV
    final_df.to_csv('final_position.csv', index=False)

    # Export android fixes DataFrame to CSV
    android_fixes.to_csv('android_position.csv', index=False)
   
    # Notice: You need to download 'Geo Data Viewer' plugin to view the KML file
    plot_position_offset(ned_df, False) # Change to True to show the plot 

    # Save KML file
    new_file_path = file_path[:-4] + "-KML.kml" # Remove .txt and add -KML.kml

    # Generate KML file with the calculated positions into the
    generate_kml_file(final_df, new_file_path)
    print(f"Successfully generated {new_file_path} file.")


# Calculate satellite position based on ephemeris data and transmit times
def calculate_satellite_position(ephemeris, transmit_time):
    mu = 3.986005e14 # Gravitational constant for Earth
    OmegaDot_e = 7.2921151467e-5 # Earth's rotation rate
    F = -4.442807633e-10 # Relativity correction coefficient

# GPS_time, SatPRN (ID), Sat.X, Sat.Y, Sat.Z, Pseudo-Range, CN0, Doppler
    sv_position = pd.DataFrame()
    sv_position['SatPRN'] = ephemeris.index
    sv_position.set_index('SatPRN', inplace=True)
    sv_position['GPS_time'] = transmit_time['tTxSeconds'] - ephemeris['t_oe']
    A = ephemeris['sqrtA'].pow(2)
    n_0 = np.sqrt(mu / A.pow(3))
    n = n_0 + ephemeris['deltaN']
    M_k = ephemeris['M_0'] + n * sv_position['GPS_time']
    E_k = M_k
    err = pd.Series(data=[1] * len(sv_position.index))
    i = 0

    # Calculate eccentric anomaly E_k using Newton's method
    while err.abs().min() > 1e-8 and i < 10:
        new_vals = M_k + ephemeris['e'] * np.sin(E_k)
        err = new_vals - E_k
        E_k = new_vals
        i += 1

    sinE_k = np.sin(E_k)
    cosE_k = np.cos(E_k)
    delT_r = F * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k
    delT_oc = transmit_time['tTxSeconds'] - ephemeris['t_oc']
    sv_position['delT_sv'] = (ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc +
                              ephemeris['SVclockDriftRate'] * delT_oc.pow(2))

    v_k = np.arctan2(np.sqrt(1 - ephemeris['e'].pow(2)) * sinE_k, (cosE_k - ephemeris['e']))
    Phi_k = v_k + ephemeris['omega']

    sin2Phi_k = np.sin(2 * Phi_k)
    cos2Phi_k = np.cos(2 * Phi_k)
    du_k = ephemeris['C_us'] * sin2Phi_k + ephemeris['C_uc'] * cos2Phi_k
    dr_k = ephemeris['C_rs'] * sin2Phi_k + ephemeris['C_rc'] * cos2Phi_k
    di_k = ephemeris['C_is'] * sin2Phi_k + ephemeris['C_ic'] * cos2Phi_k

    u_k = Phi_k + du_k
    r_k = A * (1 - ephemeris['e'] * np.cos(E_k)) + dr_k
    i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT'] * sv_position['GPS_time']

    x_k_prime = r_k * np.cos(u_k)
    y_k_prime = r_k * np.sin(u_k)
    Omega_k = (ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e) * sv_position['GPS_time'] -
               OmegaDot_e * ephemeris['t_oe'])

    sv_position['Sat_X'] = (x_k_prime * np.cos(Omega_k) -
                          y_k_prime * np.cos(i_k) * np.sin(Omega_k))
    sv_position['Sat_Y'] = (x_k_prime * np.sin(Omega_k) +
                          y_k_prime * np.cos(i_k) * np.cos(Omega_k))
    sv_position['Sat_Z'] = y_k_prime * np.sin(i_k)
    sv_position['CN0'] = transmit_time['Cn0DbHz'] # Signal strength

    return sv_position

# Perform a least squares adjustment to estimate the receiver's position and clock bias
def least_squares(xs, measured_pseudorange, x0, b0):
    dx = 100 * np.ones(3)
    b = b0
    G = np.ones((measured_pseudorange.size, 4))
    iterations = 0

    while np.linalg.norm(dx) > 1e-3:
        r = np.linalg.norm(xs - x0, axis=1)
        phat = r + b0
        deltaP = measured_pseudorange - phat
        G[:, 0:3] = -(xs - x0) / r[:, None]
        sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ deltaP
        dx = sol[0:3]
        db = sol[3]
        x0 = x0 + dx
        b0 = b0 + db

    norm_dp = np.linalg.norm(deltaP)
    return x0, b0, norm_dp


def generate_kml_file(final_df, new_file_path):
    # Generate KML file
    kml = simplekml.Kml()
    index = 0
    while index < len(final_df):
        row = final_df.iloc[index]
        kml.newpoint(name=str(index), coords=[(row['Longitude'], row['Latitude'], row['Altitude'])])
        index += 1

    kml.save(new_file_path)

def parse_and_format_file(file_path):
    # Parse GNSS data from the log file into pandas DataFrames (tables)
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0][0] == '#':  # Skip comments
                if 'Fix' in row[0]:  # Check if the row contains the header for the fix data
                    android_fixes = [row[1:]]
                elif 'Raw' in row[0]:
                    measurements = [row[1:]]
            else:
                if row[0] == 'Fix':
                    android_fixes.append(row[1:])
                elif row[0] == 'Raw':
                    measurements.append(row[1:])

    # Convert lists to DataFrames for easier manipulation
    android_fixes = pd.DataFrame(android_fixes[1:], columns=android_fixes[0])
    measurements = pd.DataFrame(measurements[1:], columns=measurements[0])

    # Format and parse measurements data
    # measurements = format_and_parse_measurements(measurements)
    
    # Format satellite IDs
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']
    measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
    measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'
    measurements['SvName'] = measurements['Constellation'] + measurements['Svid']

    # Remove all non-GPS measurements
    measurements = measurements.loc[measurements['Constellation'] == 'G']

    # Convert columns to numeric representation
    numeric_columns = ['Cn0DbHz', 'TimeNanos', 'FullBiasNanos', 'ReceivedSvTimeNanos', 'PseudorangeRateMetersPerSecond',
                       'ReceivedSvTimeUncertaintyNanos']

    index = 0
    while index < len(numeric_columns):
        col = numeric_columns[index]
        measurements[col] = pd.to_numeric(measurements[col])
        index += 1

    # Initialize missing measurement values
    if 'BiasNanos' not in measurements.columns:
        measurements['BiasNanos'] = 0
    else:
        measurements['BiasNanos'] = pd.to_numeric(measurements['BiasNanos'])

    if 'TimeOffsetNanos' not in measurements.columns:
        measurements['TimeOffsetNanos'] = 0
    else:
        measurements['TimeOffsetNanos'] = pd.to_numeric(measurements['TimeOffsetNanos'])

    # Convert GNSS time to Unix time
    measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (measurements['FullBiasNanos'] - measurements['BiasNanos'])
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
    measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc=True, origin=gpsepoch)

    # Split data into measurement epochs
    measurements['Epoch'] = 0
    measurements.loc[measurements['UnixTime'] - measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    measurements['Epoch'] = measurements['Epoch'].cumsum()

    # Calculate pseudoranges
    measurements['tRxGnssNanos'] = (measurements['TimeNanos'] + measurements['TimeOffsetNanos'] -
                                    (measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0]))
    measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
    measurements['tRxSeconds'] = 1e-9 * measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
    measurements['tTxSeconds'] = 1e-9 * (measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])
    measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']
    measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
    measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']

    return android_fixes, measurements


def calculate_ecef_coordinates(measurements, manager, sv_position, timestamp, sats):
    ecef_list = []
    for epoch in measurements['Epoch'].unique():
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)]
        one_epoch = one_epoch.drop_duplicates(subset='SvName').set_index('SvName')
        if len(one_epoch.index) > 4:
            timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
            sats = one_epoch.index.unique().tolist()
            ephemeris = manager.get_ephemeris(timestamp, sats)
            sv_position = calculate_satellite_position(ephemeris, one_epoch)

            xs = sv_position[['Sat_X', 'Sat_Y', 'Sat_Z']].to_numpy()
            pr = one_epoch['PrM'] + LIGHTSPEED * sv_position['delT_sv']
            pr = pr.to_numpy()

            x = 0
            b = 0
            dp = 0
            x, b, dp = least_squares(xs, pr, x, b)
            ecef_list.append(x)

    return ecef_list


def plot_position_offset(ned_df, show):
    plt.style.use('dark_background')
    plt.plot(ned_df['E'], ned_df['N'])
    plt.title('Position Offset From First Epoch')
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.gca().set_aspect('equal', adjustable='box')
    if (show):
        plt.show()
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GNSS data from a specified file.")
    parser.add_argument('file_path', type=str, help='Path to the GNSS data file')
    args = parser.parse_args()
    main(args.file_path)
