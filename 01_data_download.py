# -*- coding: utf-8 -*-
"""
01_download_data.py

Downloads raw infrasound and seismic waveform data from IRIS/EarthScope
for the Barry Arm, Alaska monitoring network. Data is saved as daily
miniSEED files, one file per station/channel/day.

Existing files are skipped automatically, so the script can be re-run
safely to fill in missing days or extend the time range.

Stations downloaded:
    BAEI  (AV network) — infrasound array, channel HDF
    BAT   (AK network) — seismic, channel BHZ
    BAE   (AK network) — seismic, channel BHZ

Outputs:
    Daily miniSEED files saved to subdirectories under OUTPUT_BASE_DIR,
    organized by station:
        <OUTPUT_BASE_DIR>/BAEI/BAEI..HDF.YYYY-MM-DDTHH-MM-SS_24h_100Hz.ms
        <OUTPUT_BASE_DIR>/BAT/BAT..BHZ.YYYY-MM-DDTHH-MM-SS_24h_100Hz.ms
        <OUTPUT_BASE_DIR>/BAE/BAE..BHZ.YYYY-MM-DDTHH-MM-SS_24h_100Hz.ms

Usage:
    Set the time range in the USER PARAMETERS section and run the script.
    To download only a subset of stations, comment out entries in the
    STATIONS list.

Dependencies:
    obspy
    See requirements.txt for version details.

"""

# ============================================================
# IMPORTS
# ============================================================

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import os

# ============================================================
# USER PARAMETERS — edit these before running
# ============================================================

# Time range to download (END date is not included)
# In format: YYYY-MM-DDTHH:MM:SS note that the 'T' should remain
START = UTCDateTime('')
END   = UTCDateTime('')

# Base directory where all downloaded data will be saved
# Each station gets its own subdirectory created automatically
OUTPUT_BASE_DIR = ''

# Stations to download — comment out any you don't need
# Each entry: (network, station, location, channel, output_dir)
# Ex: ('AV', 'BAEI', '*', 'HDF', r'C:\...\data\raw\infrasound')
STATIONS = [
    ('', '', '', '', r''),
    ('', '', '', '', r''),
    ('', '', '', '', r''),
]

# ============================================================
# FUNCTIONS
# ============================================================


def download_station(client, net, sta, loc, cha, start, end, output_base_dir):
    """
    Download daily miniSEED files for one station over a date range.

    Files are named: STA.LOC.CHA.YYYY-MM-DDTHH-MM-SS_24h_100Hz.ms
    and saved to: output_base_dir/STA/

    Skips days where the file already exists. Skips days where IRIS
    returns no data.

    Parameters
    ----------
    client : obspy.clients.fdsn.Client
        Initialized FDSN client.
    net : str
        Network code (e.g., 'AV', 'AK').
    sta : str
        Station code (e.g., 'BAEI', 'BAT').
    loc : str
        Location code. Use '*' for wildcard, '' for empty.
    cha : str
        Channel code (e.g., 'HDF', 'BHZ').
    start : UTCDateTime
        Start of the download window.
    end : UTCDateTime
        End of the download window (not included).
    output_base_dir : str
        Root directory under which a per-station subdirectory is created.
    """

    # Create output directory for this station
    sta_dir = os.path.join(output_base_dir, sta)
    os.makedirs(sta_dir, exist_ok=True)

    num_days  = int((end - start) / 86400)
    loc_label = loc if loc not in ('', '*') else ''

    print(f'\n{"="*60}')
    print(f'Downloading {net}.{sta}.{loc}.{cha}')
    print(f'Period: {start.date} to {end.date} ({num_days} days)')
    print(f'Output: {sta_dir}')
    print(f'{"="*60}')

    for n in range(num_days):

        t1 = start + (n * 86400)
        t2 = t1 + 86400

        date_str = (f'{t1.year}-{t1.month:02d}-{t1.day:02d}'
                    f'T{t1.hour:02d}-{t1.minute:02d}-{t1.second:02d}')
        filename = f'{sta}.{loc_label}.{cha}.{date_str}_24h_100Hz.ms'
        filepath = os.path.join(sta_dir, filename)

        print(f'\n  Day {n+1}/{num_days}: {t1.date}')

        # Skip if file already exists
        if os.path.exists(filepath):
            print(f'  File already exists, skipping.')
            continue

        # Download from IRIS
        try:
            st = client.get_waveforms(net, sta, loc, cha, t1, t2,
                                      attach_response=True)
            st.remove_sensitivity()
        except Exception as e:
            print(f'  No data available: {e}')
            continue

        # Basic preprocessing before saving
        st.detrend('demean').taper(0.05)
        st.merge(method=1, fill_value=0)

        # Save as miniSEED
        st.write(filepath, format='MSEED')
        print(f'  Saved: {filename}')

    print(f'\nFinished {net}.{sta}.{loc}.{cha}.\n')


# ============================================================
# MAIN
# ============================================================

client = Client('IRIS')

for (net, sta, loc, cha, out_dir) in STATIONS:
    download_station(client, net, sta, loc, cha, START, END, out_dir)

print('All downloads complete.')
