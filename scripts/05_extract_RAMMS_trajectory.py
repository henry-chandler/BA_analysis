# -*- coding: utf-8 -*-
"""
05_extract_ramms_trajectories.py

Extracts key geometric properties from RAMMS::Rockfall output trajectory
shapefiles and computes the back azimuth of each simulated rockfall as seen
from the BAEI infrasound array. Results are saved as a CSV for use in
downstream analysis and comparison with observed infrasound back azimuths.

For each trajectory the following are extracted:
    start_x: UTM easting of the trajectory start point [m]
    start_y: UTM northing of the trajectory start point [m]
    length: total trajectory length [m]
    back_azimuth_deg: back azimuth from the BAEI array to the start point
                       (degrees clockwise from north, 0–360)

Coordinate system:
    All coordinates are projected to UTM Zone 6N (EPSG:32606), which covers
    the Barry Arm study area.

Inputs:
    RAMMS::Rockfall trajectory shapefile:
        <RAMMS_DIR>/BA_rockfall_Trajectories.shp

Outputs:
    CSV of per-trajectory geometry and back azimuth:
        <RAMMS_DIR>/BA_rockfall_trajectory_data.csv

Dependencies:
    geopandas, pandas, numpy, pyproj
    See requirements.txt for version details.

References:
    Leine, R.I., Schweizer, A., Christen, M., Glover, J., Bartelt, P., &
        Gerber, W. (2014). Simulation of rockfall trajectories with
        consideration of rock shape. Multibody System Dynamics, 32(2),
        241–271. https://doi.org/10.1007/s11044-013-9393-4

Author: [Your name]
"""

# ============================================================
# IMPORTS
# ============================================================

import geopandas as gpd
import numpy as np
from pyproj import Transformer
import os

# ============================================================
# USER PARAMETERS — edit these before running
# ============================================================

# Directory containing the RAMMS output shapefile
RAMMS_DIR = ''

# RAMMS trajectory shapefile filename
SHAPEFILE = ''

# Output trajectory CSV filename
OUTPUT_CSV = ''

# BAEI infrasound array location (lon, lat — WGS84)
ARRAY_LON = -148.122216
ARRAY_LAT = 61.132128

# Target projected CRS — UTM Zone 6N covers the Barry Arm study area
TARGET_CRS = ''

# ============================================================
# FUNCTIONS
# ============================================================

def compute_back_azimuth(start_x, start_y, array_x, array_y):
    """
    Compute the back azimuth from the infrasound array to a trajectory
    start point, in degrees clockwise from north (0–360).

    Parameters
    ----------
    start_x : float
        UTM easting of the trajectory start point [m].
    start_y : float
        UTM northing of the trajectory start point [m].
    array_x : float
        UTM easting of the array [m].
    array_y : float
        UTM northing of the array [m].

    Returns
    -------
    float
        Back azimuth in degrees (0–360).
    """
    dx = start_x - array_x  # positive = event is east of array
    dy = start_y - array_y  # positive = event is north of array
    angle = np.degrees(np.arctan2(dx, dy))  # clockwise from north
    return angle % 360


# ============================================================
# MAIN
# ============================================================

# Build full paths
shapefile_path = os.path.join(RAMMS_DIR, SHAPEFILE)
output_path    = os.path.join(RAMMS_DIR, OUTPUT_CSV)

# Load trajectory shapefile
print(f'Loading shapefile: {shapefile_path}')
gdf = gpd.read_file(shapefile_path)
print(f'  {len(gdf)} trajectories loaded.')

# Convert array coordinates from WGS84 (lon/lat) to UTM Zone 6N
transformer = Transformer.from_crs('EPSG:4326', TARGET_CRS, always_xy=True)
array_x, array_y = transformer.transform(ARRAY_LON, ARRAY_LAT)
print(f'\nArray UTM position (EPSG:32606):')
print(f'Easting:  {array_x:.2f} m')
print(f'Northing: {array_y:.2f} m')

# Extract start point, trajectory length, and back azimuth for each trajectory
gdf['start_x'] = gdf.geometry.apply(lambda g: list(g.coords)[0][0])
gdf['start_y'] = gdf.geometry.apply(lambda g: list(g.coords)[0][1])
gdf['trajectory_length'] = gdf.geometry.length
gdf['initial_baz'] = gdf.apply(
    lambda row: compute_back_azimuth(
        row['start_x'], row['start_y'], array_x, array_y),
    axis=1
)

# Print summary
print(f'\nTrajectory summary:')
print(gdf[['start_x', 'start_y', 'trajectory_length', 'initial_baz']].to_string())
print(f'\nBack azimuth range: '
      f'{gdf["initial_baz"].min():.1f}° – '
      f'{gdf["initial_baz"].max():.1f}°')
print(f'Trajectory length range: '
      f'{gdf["trajectory_length"].min():.1f} m – '
      f'{gdf["trajectory_length"].max():.1f} m')

# Save to CSV
gdf[['start_x', 'start_y', 'trajectory_length', 'initial_baz']].to_csv(
    output_path, index=False
)
print(f'\nResults saved to: {output_path}')