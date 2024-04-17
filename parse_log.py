import re
import numpy as np
import pandas as pd
import os

#part 1
def parse_log_to_csv(log_filename, csv_filename):
    with open(log_filename, 'r') as file:
        lines = file.readlines()

    csv_data = []
    csv_data.append("GPS time,SatPRN (ID),Sat.X,Sat.Y,Sat.Z,Pseudo-Range,CN0,Doppler")

    for line in lines:
        if line.startswith("Status"):
            parts = line.split(',')
            gps_time = parts[6]  # UnixTimeMillis
            sat_prn = parts[5]   # Svid
            cn0 = parts[7]       # Cn0DbHz
            doppler = "N/A"      # Placeholder as no direct Doppler shift data is given

            # Adding a line in CSV format
            csv_data.append(f"{gps_time},{sat_prn},N/A,N/A,N/A,N/A,{cn0},{doppler}")

    # Write the CSV data to a file
    with open(csv_filename, 'w') as file:
        for entry in csv_data:
            file.write(entry + "\n")

parse_log_to_csv("driving.txt", "data.csv")
current_directory = os.path.dirname(__file__)
file_name = 'data.csv'
file_path = os.path.join(current_directory, file_name)
df = pd.read_csv(file_path)


#function of identify 5 most useful
data='data.csv'
def find_strongest_sats(data, n_sats=5):
  # מיון נתונים לפי CN0 יורד
  sorted_data = data[data[:, -1].argsort()][-n_sats:]

  # ראשיון SatPRNs ייחודיים
  unique_sats = []
  for row in sorted_data:
    if row[1] not in unique_sats:
      unique_sats.append(row[1])

  # איטרציה עד לקבלת 5 לווינים
  while len(unique_sats) < n_sats:
    # בחירת הערך החזק הבא (שאינו ייחודי)
    next_strongest = sorted_data[sorted_data[:, 1] == unique_sats[-1]][0, 1]
    if next_strongest not in unique_sats:
      unique_sats.append(next_strongest)

  # אילוץ אורך רשימה
  strongest_sats = unique_sats[:n_sats]
  return strongest_sats

#function of identify x,y,z,pseudo-range
def rms_positioning(data, gps_time, sat_mask, r_weights):
 
  selected_data = data[data[:, 0] == gps_time, :]
  selected_data = selected_data[sat_mask == 1, :]
  num_sats = selected_data.shape[0]
  if num_sats < 4:
    return None

  x0 = np.mean(selected_data[:, 2])  # Sat.X
  y0 = np.mean(selected_data[:, 3])  # Sat.Y
  z0 = np.mean(selected_data[:, 4])  # Sat.Z

  max_iter = 100
  tol = 1e-6

  for _ in range(max_iter):
    # calcute the range of the value
    r_est = np.sqrt((selected_data[:, 2] - x0)**2 +
                     (selected_data[:, 3] - y0)**2 +
                     (selected_data[:, 4] - z0)**2)
    
    v = (r_est - selected_data[:, 5]) / r_est
    w_new = r_weights / (1 + v**2)

    # calculate update location
    delta_x = np.sum(w_new * (selected_data[:, 2] - x0))
    delta_y = np.sum(w_new * (selected_data[:, 3] - y0))
    delta_z = np.sum(w_new * (selected_data[:, 4] - z0))

   # update location
    x1 = x0 + delta_x
    y1 = y0 + delta_y
    z1 = z0 + delta_z

    # convergence test
    if np.linalg.norm([delta_x, delta_y, delta_z]) < tol:
      break

    # update vareible
    x0 = x1
    y0 = y1
    z0 = z1

  # where the location
  pos_est = np.array([x1, y1, z1])

  return pos_est

# data = np.loadtxt("data.csv", delimiter=",") 
gps_time = (df['GPS time'])
strongest_sats = find_strongest_sats(data)
print(f"most 5 powerfull  satellites {strongest_sats}")

sat_mask = strongest_sats 
r_weights =  df['CN0'] 
pos_est = rms_positioning(data, gps_time, sat_mask, r_weights)
print(f"location: {pos_est}") 
df['Sat.X'],df['Sat.Y'],df['Sat.Z']= rms_positioning(data, gps_time, sat_mask, r_weights)

df['Sat.X'] = pd.to_numeric(df['Sat.X'])
df['Sat.Y'] = pd.to_numeric(df['Sat.Y'])
df['Sat.Z'] = pd.to_numeric(df['Sat.Z'])
df['Pseudo-Range'] = pd.to_numeric(df['Pseudo-Range'])

def ecef_to_lla(ecef ):
    a=6371000
    f=0.0335281066
    x, y, z = ecef
    r = np.linalg.norm(ecef)
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    lon = np.arctan2(y, x)
    h = r - a * (1 - f**2) * np.sin(lat)**2 / np.sqrt(1 - f**2 * np.sin(lat)**2)

    return np.array([lat, lon, h])

ecef = np.array([5962938.67720026, 1666406.33860679, 4162276.40918602])  
lla = ecef_to_lla(ecef)
print(f"geographic coordinates: {lla}")
df['Sat.X'],df['Sat.Y'], df['Sat.Z']= ecef_to_lla(df['Sat.X']),(df['Sat.Y']),(df['Sat.Z'])
print(df.head)
#position = gps_positioning("data.csv", 1575420030)
#print(f"Computed Position: {position}")
