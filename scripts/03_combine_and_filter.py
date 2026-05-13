# -*- coding: utf-8 -*-
"""
03_combine_and_filter_events.py

Loads processed infrasound event results from multiple years, combines them
into a single dataset, and applies two filters:

    1. Snow cover mask  — removes events during periods of full snow cover
                          (November 9 through March 20), when rockfall signals
                          are obscured or suppressed by snow.
    2. Back azimuth mask — retains only events within the target back azimuth
                           range (250–305 degrees) pointing toward Barry Arm.

Produces a diagnostic scatter plot of all events before filtering, then saves
the filtered combined dataset as a single .npy file for use in downstream
analysis.

Inputs:
    Processed event ID .npy files from 01_array_processing.py, one per year.

Outputs:
    figures/BAEI_events_all_years_prefilt.png  — scatter plot before filtering
    data/processed/BAEI_events_YYYY-YYYY.npy   — filtered combined event dataset

Dependencies:
    numpy, pandas, matplotlib, obspy
    See requirements.txt for version details.

"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
from datetime import datetime

# ============================================================
# USER PARAMETERS — edit these before running
# ============================================================

# Base directory for processed infrasound event ID results
INFRASOUND_DIR = ''

# Output directories
PROCESSED_DIR = ''
FIGURES_DIR = ''

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR,   exist_ok=True)

# Per-year event ID result files — comment out years not yet processed
# These should be the event_id files that were output from script 02_array_processing
YEAR_FILES = {
    2022: os.path.join(''),
    2023: os.path.join(''),
    2024: os.path.join(''),
    2025: os.path.join('')
}

# Back azimuth filter bounds [degrees] — should match 01_array_processing.py
BAZ_MIN = 250
BAZ_MAX = 305

# Snow cover window (month, day) — applied every year in the dataset
# Events between SNOW_START and SNOW_END (inclusive) are removed
SNOW_START = (11, 9)   # November 9
SNOW_END = (3, 20)   # March 20

# Output filename label (auto-generated from year range if left as None)
OUTPUT_LABEL = None

# ============================================================
# FUNCTIONS
# ============================================================

def load_year(filepath, year):
    """
    Load one year of processed event ID results from a .npy file.

    Parameters
    ----------
    filepath : str
        Path to the .npy event ID file.
    year : int
        Year label, used only for print output.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: yearly_event_time, yearly_event_dur, yearly_initial_baz. 
        Returns None if the file is not found.
    """
    if not os.path.exists(filepath):
        print(f'  WARNING: File not found for {year}, skipping: {filepath}')
        return None

    data = np.load(filepath, allow_pickle=True).item()

    df = pd.DataFrame({
        'yearly_event_times': data['event_times'],
        'yearly_event_dur': data['event_dur'],
        'yearly_initial_baz': data['initial_baz'],
    })

    print(f'  {year}: {len(df)} events loaded.')
    return df


def is_snow_covered(date, snow_start, snow_end):
    """
    Return True if a date falls within the annual snow cover window.

    The snow cover window spans from snow_start (month, day) through
    snow_end (month, day) of the following calendar year, wrapping
    across the year boundary.

    Parameters
    ----------
    date : datetime
        Date to check.
    snow_start : tuple of (int, int)
        (month, day) marking the start of snow cover.
    snow_end : tuple of (int, int)
        (month, day) marking the end of snow cover.

    Returns
    -------
    bool
    """
    m, d = date.month, date.day
    start_m, start_d = snow_start
    end_m, end_d = snow_end

    after_start = (m > start_m) or (m == start_m and d >= start_d)
    before_end = (m < end_m) or (m == end_m and d <= end_d)

    # Window wraps across year boundary (e.g. Nov → Mar)
    return after_start or before_end


def plot_prefilt(event_times_dt, initial_baz, event_dur, figures_dir):
    """
    Scatter plot of all events before any filtering, colored by event duration.
    Saved to figures_dir.

    Parameters
    ----------
    event_times_dt : np.ndarray of datetime
        Event times as Python datetimes.
    initial_baz : np.ndarray
        Initial back azimuths [degrees].
    event_dur : np.ndarray
        Event durations [s].
    figures_dir : str
        Directory in which to save the figure.
    """
    norm = mcolors.Normalize(vmin=16, vmax=50)
    fig, ax = plt.subplots(figsize=(16, 6))
    sc = ax.scatter(event_times_dt, initial_baz, c=event_dur,
                    cmap='viridis', norm=norm, s=7)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Event Duration (s)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Initial Back Azimuth (°)')
    ax.set_title('All events before snow and back azimuth filtering')
    ax.set_xlim(right=datetime(event_times_dt[-1].year, 12, 31))
    plt.tight_layout()

    outpath = os.path.join(figures_dir, 'BAEI_events_all_years_prefilt.png')
    fig.savefig(outpath, dpi=200)
    print(f'Pre-filter plot saved to: {outpath}')
    plt.show()


# ============================================================
# MAIN
# ============================================================

# --- Load and combine all years ---

print('Loading event ID results...')
frames = []
for year, filepath in YEAR_FILES.items():
    df = load_year(filepath, year)
    if df is not None:
        frames.append(df)

if len(frames) == 0:
    raise FileNotFoundError('No event files were found. Check YEAR_FILES paths.')

df_all = pd.concat(frames, ignore_index=True)
print(f'\nTotal events before filtering: {len(df_all)}')

# Extract arrays
event_times = np.array(df_all['yearly_event_times'])
event_dur = np.array(df_all['yearly_event_dur'])
initial_baz = np.array(df_all['yearly_initial_baz'])

# Convert UTCDateTime to Python datetime for masking and plotting
event_times_dt = np.array([t.datetime for t in event_times])

# --- Pre-filter diagnostic plot ---

plot_prefilt(event_times_dt, initial_baz, event_dur, FIGURES_DIR)

# --- Apply snow cover mask ---

snow_mask = np.array([is_snow_covered(t, SNOW_START, SNOW_END) for t in event_times_dt])
event_times = event_times[~snow_mask]
event_times_dt = event_times_dt[~snow_mask]
event_dur = event_dur[~snow_mask]
initial_baz = initial_baz[~snow_mask]
print(f'Events after snow cover mask: {len(event_times)}'
      f'({snow_mask.sum()} removed)')

# --- Apply back azimuth mask ---

baz_mask = (initial_baz >= BAZ_MIN) & (initial_baz <= BAZ_MAX)
event_times = event_times[baz_mask]
event_times_dt = event_times_dt[baz_mask]
event_dur = event_dur[baz_mask]
initial_baz = initial_baz[baz_mask]
print(f'Events after back azimuth mask: {len(event_times)}'
      f'({(~baz_mask).sum()} removed, kept {BAZ_MIN}–{BAZ_MAX}°)')

# --- Save filtered dataset ---

all_events = {
    'event_times': event_times,
    'event_dur': event_dur,
    'initial_baz': initial_baz,
}

years      = sorted(YEAR_FILES.keys())
label      = OUTPUT_LABEL or f'BAEI_events_{years[0]}-{years[-1]}'
save_path  = os.path.join(PROCESSED_DIR, label)
np.save(save_path, all_events)
print(f'\nFiltered events saved to: {save_path}.npy')
print(f'Final event count: {len(event_times)}')
