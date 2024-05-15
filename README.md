# GNSS Data Processing and Position Calculation

This repository contains a Python script that processes GNSS data, calculates satellite positions, converts coordinates, and generates output files including KML for visualization. The script uses various libraries for data handling, mathematical computations, and plotting.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Functions](#functions)
- [Example](#example)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/roy-asraf1/Autonomous-robots.git
    cd Autonomous-robots
    ```

2. Install the required Python packages (Specific versions):
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your GNSS data log file with the required format (`.txt`).
2. Run the script with your data file as an argument:
    ```bash
    python task_code.py <file_path>
    ```

## Dependencies

The script requires the following Python packages:

- sys
- os
- csv
- simplekml
- datetime
- pandas
- numpy
- navpy
- gnssutils
- matplotlib

Make sure to install these packages using the `requirements.txt` file.

## Functions

### `main(file_path)`
Main function to process the GNSS data file and generate output files.

### `parse_and_format_file(file_path)`
Parses the GNSS data file and formats the data into pandas DataFrames.

### `calculate_satellite_position(ephemeris, transmit_time)`
Calculates satellite positions based on ephemeris data and transmit times.

### `calculate_ecef_coordinates(measurements, manager, sv_position, timestamp, sats)`
Calculates ECEF coordinates from the GNSS measurements.

### `least_squares(xs, measured_pseudorange, x0, b0)`
Performs a least squares adjustment to estimate the receiver's position and clock bias.

### `generate_kml_file(final_df, new_file_path)`
Generates a KML file for visualization in Google Earth or other geospatial software.

### `plot_position_offset(ned_df, show)`
Plots the position offset.

## Example

An example of running the script:
```bash
python3 task_code.py driving.txt
```


This will process the data in driving.txt and output several CSV files and a KML file:

- satellite_positions_export.csv
- calculated_position_lla.csv
- final_position.csv
- android_position.csv

and also the file of the file_path name, in this case:
- driving-KML.kml

but in general:
- {file_name}-KML.kml


### Notice
In the task_code.py there is a function `plot_position_offset(ned_df, show)`
and as it's argument you can change to 'True' to get the visualization inside the IDE. (Read the comment for this in the code) 


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Contributors:
Elor Israeli, Roy Asraf