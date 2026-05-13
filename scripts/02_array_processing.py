# -*- coding: utf-8 -*-
"""
02_array_processing.py

Performs least-trimmed squares infrasound array processing on data from the
BAEI infrasound array at Barry Arm, Alaska. Detects and classifies rockfall
events based on trace velocity, back azimuth, and cross-correlation thresholds.

Inputs:
    Raw infrasound miniSEED files, one per day, expected at:
        data/raw/BAEI..HDF.YYYY-MM-DDTOO-00-00_24h_100Hz.ms

Outputs:
    - Per-event diagnostic plots saved to:        figures/events/
    - LTSVA results (.npy) saved to:              data/processed/
    - Event ID results (.npy) saved to:           data/processed/

Dependencies:
    obspy, lts_array, numpy, scipy, matplotlib, pandas
    See requirements.txt for version details.

Usage:
    Set the analysis parameters in the USER PARAMETERS section below,
    then run the script top to bottom. Data is processed one day at a time.

References:
    Bishop et al. (2020). lts_array.
    https://github.com/uafgeotools/lts_array
    
"""

# ============================================================
# IMPORTS
# ============================================================

from obspy import UTCDateTime, Stream, read
from lts_array import ltsva
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from scipy.signal import welch
from matplotlib.colors import Normalize
import pandas as pd

# ============================================================
# USER PARAMETERS — edit these before running
# ============================================================

# Station and channel
STATION = 'BAEI'
CHANNEL = 'HDF'

# Analysis time window (END date is not included in the run)
# In format: YYYY-MM-DDTHH:MM:SS note that the 'T' should remain
START = UTCDateTime('')
END   = UTCDateTime('')

# Array processing parameters
FMIN = 1    # Lower frequency bound [Hz]
FMAX = 30   # Upper frequency bound [Hz]
WINLEN = 8    # Window length [s]
WINOVER = 0.5  # Window overlap [proportion]
ALPHA = 1.0  # LTS fraction (1.0 = ordinary least squares)

# Event detection thresholds (apply a wider baz range than necessary, can be trimmed at a later stage)
CORR_THRESH = 0.8   # Minimum median cross-correlation to flag an event
EVAL_LENGTH = 3     # Minimum number of consecutive windows above threshold
BAZ_MIN = 220   # Minimum back azimuth [degrees]
BAZ_MAX = 340   # Maximum back azimuth [degrees]

# Sensor coordinates (sensor index: (lat, lon))
COORDS = {
    1: (61.132675, -148.1219),
    2: (61.132085, -148.12107),
    3: (61.13173, -148.121985),
    4: (61.131875, -148.122915),
    5: (61.132405, -148.12281),
    6: (61.13218, -148.12206)
}

# Paths — relative to the repository root
# RAW_INF_DATA_DIR should be the directory in which the infrasound data was saved from script 01_data_download
RAW_INF_DATA_DIR = ''
PROCESSED_DIR = ''
FIGURES_DIR = ''
# Create output directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR,   exist_ok=True)

# ============================================================
# FUNCTIONS
# ============================================================

def event_id(st, START, mdccm, baz, vel, WINLEN, WINOVER,
             corr_thresh, eval_length, baz_min, baz_max):
    """
    Identify rockfall events from LTSVA output based on correlation,
    back azimuth, and trace velocity thresh0olds.

    Parameters
    ----------
    st : obspy.Stream
        Raw waveform data.
    START : UTCDateTime
        Start time of the analysis window.
    mdccm : np.ndarray
        Median cross-correlation maxima for each window.
    baz : np.ndarray
        Back azimuth [degrees] for each window.
    vel : np.ndarray
        Trace velocity [km/s] for each window.
    WINLEN : float
        Window length [s].
    WINOVER : float
        Window overlap [proportion].
    corr_thresh : float
        Minimum correlation to flag a window as an event.
    eval_length : int
        Number of consecutive windows that must exceed corr_thresh.
    baz_min : float
        Lower back azimuth bound [degrees].
    baz_max : float
        Upper back azimuth bound [degrees].

    Returns
    -------
    daily_event_idx : np.ndarray
        Indices of event start windows (after all filtering).
    daily_event_times : np.ndarray
        UTC start times of detected events.
    daily_event_dur_win : np.ndarray
        Duration of each event in windows.
    daily_event_dur : np.ndarray
        Duration of each event in seconds.
    daily_initial_baz : np.ndarray
        Initial back azimuth of each event.
    """

    # Initialize output arrays
    daily_event_idx = []
    daily_event_times = []
    daily_event_dur_win = []
    daily_initial_baz = []

    i = 0
    while i < len(mdccm):

        # Define the evaluation window, clamping at array end
        if i < (len(mdccm) - eval_length):
            index_range = range(i, i + eval_length)
        else:
            index_range = range(i, len(mdccm))

        # Skip windows with NaN values
        if math.isnan(vel[i]) or math.isnan(baz[i]) or math.isnan(mdccm[i]):
            i += 1
            continue

        # Check that all windows in the evaluation range exceed the threshold
        if np.all(mdccm[index_range] >= corr_thresh):

            # Apply back azimuth filter
            if baz_min < baz[i] < baz_max:

                # Determine event duration by stepping forward until correlation drops
                j = 0
                while mdccm[i + j] >= corr_thresh and (i + j) < (len(mdccm) - 1):
                    j += 1

                daily_event_dur_win.append(j)
                daily_event_idx.append(i)
                daily_event_times.append(
                    START + ((i - 1) * (WINLEN * (1 - WINOVER)) + WINLEN)
                )
                daily_initial_baz.append(baz[i])

                # Skip forward to avoid double-counting the same event
                seconds_to_skip = 120
                i = i + int(1 + ((seconds_to_skip - WINLEN) / (WINLEN * (1 - WINOVER))))
                continue

        i += 1

    # Convert lists to arrays
    daily_event_idx = np.array(daily_event_idx, dtype=int)
    daily_event_times = np.array(daily_event_times)
    daily_event_dur_win = np.array(daily_event_dur_win, dtype=int)
    daily_event_dur = ((daily_event_dur_win - 1) * (WINLEN * (1 - WINOVER)) + WINLEN).astype(int)
    daily_initial_baz = np.array(daily_initial_baz)

    # Secondary filter: velocity mean/std and back azimuth stability
    if len(daily_event_idx) > 0:
        keep_mask = []
        for i in range(len(daily_event_idx)):
            event_range = range(daily_event_idx[i], daily_event_idx[i] + daily_event_dur_win[i])
            mean_vel = np.nanmean(vel[event_range])
            std_vel  = np.nanstd(vel[event_range])
            # Require stable back azimuth (no jump > 30 degrees between windows)
            baz_stable = not np.any(np.abs(np.diff(baz[event_range])) > 30)
            keep_mask.append(
                (std_vel <= 0.01) and (0.3 <= mean_vel <= 0.4) and baz_stable
            )

        keep_mask = np.array(keep_mask)
        daily_event_idx = daily_event_idx[keep_mask]
        daily_event_times = daily_event_times[keep_mask]
        daily_event_dur_win = daily_event_dur_win[keep_mask]
        daily_event_dur = daily_event_dur[keep_mask]
        daily_initial_baz = daily_initial_baz[keep_mask]

    print(
        f'\nLTSVA and event ID complete.\n'
        f'-> {len(daily_event_idx)} events after back azimuth and velocity filter '
        f'({baz_min}–{baz_max} deg).\n'
    )

    return (daily_event_idx, daily_event_times, daily_event_dur_win, daily_event_dur,
            daily_initial_baz)


def plot_save(st, START, mdccm, baz, t, vel, sigma_tau,
              WINLEN, WINOVER, daily_event_idx, daily_event_times,
              daily_event_dur_win, daily_event_dur, daily_initial_baz,
              plot_save_dir=None):
    """
    Generate and optionally save a diagnostic figure for each detected event.
    Each figure contains: waveform, MdCCM, trace velocity, back azimuth,
    spectrogram, and power spectral density.

    Parameters
    ----------
    st : obspy.Stream
        Raw waveform data.
    START : UTCDateTime
        Start time of the analysis window.
    mdccm : np.ndarray
        Median cross-correlation maxima.
    baz : np.ndarray
        Back azimuth [degrees].
    t : np.ndarray
        Time vector (matplotlib float format).
    vel : np.ndarray
        Trace velocity [km/s].
    sigma_tau : np.ndarray
        Uncertainty in lag times.
    WINLEN : float
        Window length [s].
    WINOVER : float
        Window overlap [proportion].
    daily_event_idx : np.ndarray
        Start indices of detected events.
    daily_event_times : np.ndarray
        UTC start times of detected events.
    daily_event_dur_win : np.ndarray
        Event durations in windows.
    daily_event_dur : np.ndarray
        Event durations in seconds.
    daily_initial_baz : np.ndarray
        Initial back azimuths.
    plot_save_dir : str, optional
        Directory to save figures. If None, figures are shown but not saved.
    """

    CMAP    = 'RdYlBu_r'
    CCM_LIM = (0.2, 1)

    for i in range(len(daily_event_idx)):

        idx_start = daily_event_idx[i] - (4 * WINLEN)
        idx_end   = daily_event_idx[i] + daily_event_dur_win[i] + (4 * WINLEN)

        if idx_start < 0 or idx_end > len(t) or idx_start >= idx_end:
            print(f'Skipping event {i+1}: index range out of bounds '
                  f'({idx_start} to {idx_end})')
            continue

        # Trim waveform traces for plotting and PSD
        tr_plot = st.copy().trim(
            starttime=daily_event_times[i] - (16 * WINLEN),
            endtime=daily_event_times[i] + daily_event_dur[i] + (15 * WINLEN)
        )
        tr_psd = st.copy().trim(
            starttime=daily_event_times[i],
            endtime=daily_event_times[i] + daily_event_dur[i]
        )
        tr_spec = st[0].copy().trim(
            starttime=daily_event_times[i] - (10 * WINLEN),
            endtime=daily_event_times[i] + daily_event_dur[i] + (9 * WINLEN)
        )

        # Slice array processing results to the plot window
        t_plot     = t[idx_start:idx_end]
        mdccm_plot = mdccm[idx_start:idx_end]
        vel_plot   = vel[idx_start:idx_end]
        baz_plot   = baz[idx_start:idx_end]
        tvec       = tr_plot[0].times('matplotlib')

        # Build figure layout
        fig_all = plt.figure(figsize=(18, 10))
        ax1 = fig_all.add_axes([0.1, 0.85, 0.4, 0.125]) # Waveform
        ax2 = fig_all.add_axes([0.1, 0.71, 0.4, 0.125], sharex=ax1) # MdCCM
        ax3 = fig_all.add_axes([0.1, 0.57, 0.4, 0.125], sharex=ax1) # Velocity
        ax4 = fig_all.add_axes([0.1, 0.42, 0.4, 0.125], sharex=ax1) # Back azimuth
        ax5 = fig_all.add_axes([0.51, 0.42, 0.02, 0.415]) # MdCCM colorbar
        ax6 = fig_all.add_axes([0.1, 0.05, 0.4,  0.3]) # Spectrogram
        ax7 = fig_all.add_axes([0.51, 0.05, 0.02, 0.3]) # Spectrogram colorbar
        ax8 = fig_all.add_axes([0.63, 0.55, 0.35, 0.425]) # PSD

        # Waveform
        ax1.plot(tvec, tr_plot[0].data, 'k')
        ax1.set_xlim(t_plot[0], t_plot[-1])
        ax1.set_ylabel('Pressure [Pa]', fontsize=14)
        ax1.tick_params(labelbottom=False)
        ax1.set_title('Signal Analyses', fontsize=14)

        # MdCCM
        sc = ax2.scatter(t_plot, mdccm_plot, c=mdccm_plot, edgecolors='k',
                         lw=0.3, cmap=CMAP)
        ax2.plot([t_plot[0], t_plot[-1]], [CORR_THRESH, CORR_THRESH], 'k--')
        ax2.set_xlim(t_plot[0], t_plot[-1])
        ax2.set_ylim(CCM_LIM)
        sc.set_clim(CCM_LIM)
        ax2.tick_params(labelbottom=False)
        ax2.set_ylabel('MdCCM', fontsize=14)
        ax2.axvline(t[daily_event_idx[i]], color='red',  alpha=0.5, linestyle='--',
                    linewidth=2, label='Event Start')
        ax2.axvline(t[daily_event_idx[i] + (daily_event_dur_win[i] - 1)], color='blue',
                    alpha=0.5, linestyle='--', linewidth=2, label='Event End')
        ax2.legend(fontsize=11)

        # Trace velocity
        event_range = range(daily_event_idx[i], daily_event_idx[i] + daily_event_dur_win[i])
        mean_vel_event = np.nanmean(vel[event_range])
        sc = ax3.scatter(t_plot, vel_plot, c=mdccm_plot, edgecolors='k',
                         lw=0.3, cmap=CMAP)
        ax3.set_ylim(mean_vel_event - 0.15, mean_vel_event + 0.15)
        ax3.set_xlim(t_plot[0], t_plot[-1])
        sc.set_clim(CCM_LIM)
        ax3.tick_params(labelbottom=False)
        ax3.set_ylabel('Trace Velocity\n[km/s]', fontsize=14)
        ax3.axvline(t[daily_event_idx[i]], color='red',  alpha=0.5,
                    linestyle='--', linewidth=2)
        ax3.axvline(t[daily_event_idx[i] + (daily_event_dur_win[i] - 1)], color='blue',
                    alpha=0.5, linestyle='--', linewidth=2)

        # Back azimuth
        sc = ax4.scatter(t_plot, baz_plot, c=mdccm_plot, edgecolors='k',
                         lw=0.3, cmap=CMAP)
        ax4.set_ylim(0, 360)
        ax4.set_xlim(t_plot[0], t_plot[-1])
        sc.set_clim(CCM_LIM)
        ax4.set_ylabel('Back-azimuth\n[deg]', fontsize=14)
        ax4.axvline(t[daily_event_idx[i]], color='red',  alpha=0.5,
                    linestyle='--', linewidth=2)
        ax4.axvline(t[daily_event_idx[i] + (daily_event_dur_win[i] - 1)], color='blue',
                    alpha=0.5, linestyle='--', linewidth=2)
        ax4.xaxis_date()
        ax4.set_xlabel('UTC Time', fontsize=14)

        # MdCCM colorbar
        hc = fig_all.colorbar(sc, cax=ax5)
        hc.set_label('MdCCM', fontsize=14)

        # Spectrogram
        tr_spec.spectrogram(wlen=1, log=False, per_lap=0.8,
                            cmap='plasma', axes=ax6)
        img = ax6.images[0]
        img.set_clim(vmin=0, vmax=tr_plot[0].data.max() / 10)
        ax6.set_ylabel('Frequency (Hz)', fontsize=14)
        ax6.set_ylim(0, 30)
        t_spec_start = START + (((daily_event_idx[i] - 1) * (WINLEN * (1 - WINOVER))
                                  + WINLEN) - (4 * WINLEN))
        ax6.set_xlabel(f'Seconds after {t_spec_start}', fontsize=14)
        cbar_spec = fig_all.colorbar(img, cax=ax7)
        cbar_spec.ax.set_ylabel('Power', fontsize=14)

        # Power spectral density
        fs = tr_psd[0].stats.sampling_rate
        frequencies, power = welch(tr_psd[0].data, fs=fs, nperseg=1024)
        ax8.semilogx(frequencies, power, label='PSD')
        ax8.set_xlim([0, 30])
        ax8.grid(True, alpha=0.5)
        ax8.set_xlabel('Frequency (Hz)', fontsize=14)
        ax8.set_ylabel('Power (Pa\u00b2/Hz)', fontsize=14)
        ax8.set_title('Power Spectral Density', fontsize=14)
        ax8.legend()

        # Compute summary statistics for annotation
        max_freq      = frequencies[np.argmax(power)]
        mean_freq     = np.nansum(frequencies * power) / np.nansum(power)
        mean_corr     = np.nanmean(mdccm[daily_event_idx[i]:(daily_event_idx[i] + daily_event_dur_win[i] - 1)])
        t_str         = daily_event_times[i]

        annotations = [
            (f'Event at: {t_str.year:04d}-{t_str.month:02d}-{t_str.day:02d}, '
             f'{t_str.hour:02d}:{t_str.minute:02d}:{t_str.second:02d} UTC', 0.40),
            ('(YYYY-MM-DD, HH:MI:SS)', 0.378),
            (f'Duration: {daily_event_dur[i]} s', 0.34),
            (f'Mean correlation: {mean_corr:.2g}', 0.26),
            (f'Peak frequency: {max_freq:.3g} Hz', 0.22),
            (f'Mean frequency: {mean_freq:.3g} Hz', 0.18),
        ]
        for text, y_pos in annotations:
            fig_all.text(0.63, y_pos, text, fontsize=14)

        # Save or display the figure
        if isinstance(plot_save_dir, str):
            filename = (f'{STATION}..{CHANNEL}.'
                        f'{t_str.year}-{t_str.month:02d}-{t_str.day:02d}-'
                        f'{t_str.hour:02d}-{t_str.minute:02d}-{t_str.second:02d}.png')
            filepath = os.path.join(plot_save_dir, filename)
            try:
                fig_all.savefig(filepath, dpi=200)
                print(f'Saved event {i+1}/{len(daily_event_idx)}: {filename}')
            except Exception as e:
                print(f'Could not save event {i+1}: {e}')

        plt.close(fig_all)

    print('All event figures complete.\n')


def process_day(looped_START, looped_END, raw_inf_data_dir, station, channel,
                coords, FMIN, FMAX, WINLEN, WINOVER, ALPHA,
                CORR_THRESH, EVAL_LENGTH, BAZ_MIN, BAZ_MAX,
                figures_dir):
    """
    Load, preprocess, and analyze one day of infrasound data.

    Parameters
    ----------
    looped_START : UTCDateTime
        Start of the day to process.
    looped_END : UTCDateTime
        End of the day to process.
    raw_inf_data_dir : str
        Path to directory containing raw miniSEED files.
    station : str
        Station code (e.g., 'BAEI').
    channel : str
        Channel code (e.g., 'HDF').
    coords : dict
        Sensor index to (lat, lon) mapping.
    FMIN, FMAX : float
        Bandpass filter bounds [Hz].
    WINLEN : float
        Window length [s].
    WINOVER : float
        Window overlap [proportion].
    ALPHA : float
        LTS fraction.
    CORR_THRESH : float
        Correlation threshold for event detection.
    EVAL_LENGTH : int
        Minimum number of consecutive windows above threshold.
    BAZ_MIN, BAZ_MAX : float
        Back azimuth filter bounds [degrees].
    figures_dir : str
        Directory in which to save per-event figures.

    Returns
    -------
    dict or None
        Dictionary of daily results, or None if the day was skipped.
    """

    date_str = (f'{looped_START.year}-{looped_START.month:02d}'
                f'-{looped_START.day:02d}')
    print(f'\n--- Processing {date_str} ---')

    # Build expected filename for this day
    dayfile = os.path.join(
        raw_inf_data_dir,
        f'{station}..{channel}.{date_str}T00-00-00_24h_100Hz.ms'
    )

    # Load data
    try:
        st = read(dayfile)
    except FileNotFoundError:
        print(f'File not found: {dayfile} — skipping.')
        return None

    # Require at least 3 sensors for array processing
    if len(st) < 3:
        print(f'Only {len(st)} sensor(s) available — skipping.')
        return None

    # Preprocess: merge, detrend, taper, filter, trim
    st.merge(method=1, fill_value='interpolate')
    st.detrend('demean').taper(0.05)
    st.sort(keys=['location'])
    st.filter('bandpass', freqmin=FMIN, freqmax=FMAX)
    st = st.copy().trim(starttime=looped_START, endtime=looped_END, pad=True)

    # Match sensor locations to coordinates by sensor index
    latlist, lonlist = [], []
    for idx_tr, tr in enumerate(st):
        sensor_idx = int(str(st[idx_tr])[9])
        lat, lon = coords[sensor_idx]
        latlist.append(lat)
        lonlist.append(lon)
    st.sort(keys=['location'])

    print(st)
    print('Running LTSVA...')

    # Run least-trimmed squares array processing
    vel, baz, t, mdccm, stdict, sigma_tau, conf_int_vel, conf_int_baz = ltsva(
        st, latlist, lonlist, WINLEN, WINOVER, ALPHA
    )

    # Identify events
    (daily_event_idx, daily_event_times, daily_event_dur_win, daily_event_dur,
     daily_initial_baz) = event_id(
        st, looped_START, mdccm, baz, vel,
        WINLEN, WINOVER, CORR_THRESH, EVAL_LENGTH, BAZ_MIN, BAZ_MAX
    )

    # Save per-event diagnostic figures
    if len(daily_event_idx) > 0:
        print(f'Saving {len(daily_event_idx)} event figure(s)...')
        plot_save(
            st, looped_START, mdccm, baz, t, vel, sigma_tau,
            WINLEN, WINOVER, daily_event_idx, daily_event_times,
            daily_event_dur_win, daily_event_dur, daily_initial_baz,
            plot_save_dir=figures_dir
        )
    else:
        print('No events detected.')

    return {
        'vel': vel.astype(np.float32),
        'baz': baz.astype(np.float32),
        'mdccm': mdccm.astype(np.float32),
        't': t.astype(np.float32),
        'daily_event_idx': daily_event_idx.astype(np.int32),
        'daily_event_times': daily_event_times,
        'daily_event_dur_win': daily_event_dur_win.astype(np.int32),
        'daily_event_dur': daily_event_dur.astype(np.int32),
        'daily_initial_baz': daily_initial_baz.astype(np.float32),
        'sigma_tau':  sigma_tau.astype(np.float32),
        'conf_int_vel': conf_int_vel.astype(np.float32),
        'conf_int_baz': conf_int_baz.astype(np.float32),
    }



# ============================================================
# MAIN — loop through each day and accumulate results
# ============================================================

num_days = int((END - START) / 86400)

# Containers for full-period results
accum = {key: [] for key in [
    'vel', 'baz', 'mdccm', 't', 'daily_event_idx', 'daily_event_times',
    'daily_event_dur_win', 'daily_event_dur', 'daily_initial_baz',
    'sigma_tau', 'conf_int_vel', 'conf_int_baz'
]}

for n in range(num_days):
    looped_START = START + (n * 86400)
    looped_END   = looped_START + 86400

    result = process_day(
        looped_START, looped_END,
        RAW_INF_DATA_DIR, STATION, CHANNEL,
        COORDS, FMIN, FMAX, WINLEN, WINOVER, ALPHA,
        CORR_THRESH, EVAL_LENGTH, BAZ_MIN, BAZ_MAX,
        FIGURES_DIR
    )

    if result is None:
        continue

    for key in accum:
        accum[key].append(result[key])

# Concatenate daily arrays into full-period arrays
for key in accum:
    accum[key] = np.concatenate(accum[key])

# Unpack
event_idx = accum['daily_event_idx']
event_times = accum['daily_event_times']
event_dur_win = accum['daily_event_dur_win']
event_dur = accum['daily_event_dur']
initial_baz = accum['daily_initial_baz']


# ============================================================
# SAVE RESULTS
# ============================================================

date_range = (f'{START.year}-{START.month:02d}-{START.day:02d}_to_'
              f'{END.year}-{END.month:02d}-{END.day:02d}')
base_name  = f'{STATION}..{CHANNEL}.{date_range}'

print('Saving LTSVA and event ID results...')

np.save(
    os.path.join(PROCESSED_DIR, f'{base_name}_event_id.npy'),
    {
        'event_idx': event_idx,
        'event_times': event_times,
        'event_dur_win': event_dur_win,
        'event_dur': event_dur,
        'initial_baz': initial_baz,
    }
)


print(f'Results saved to {PROCESSED_DIR}/')


# ============================================================
# PRINT RUN SUMMARY
# ============================================================

print(
    f''
    f'Analysis complete: {START.year}-{START.month:02d}-{START.day:02d} '
    f'to {END.year}-{END.month:02d}-{END.day:02d}\n'
    f'--- Parameters ---\n'
    f'  Correlation threshold: {CORR_THRESH}\n'
    f'  Window length: {WINLEN} s\n'
    f'  Window overlap: {WINOVER}\n'
    f'  Evaluation length: {EVAL_LENGTH} windows '
    f'({int((EVAL_LENGTH-1)*(WINLEN*(1-WINOVER))+WINLEN)} s)\n'
    f'  Back azimuth range: {BAZ_MIN}–{BAZ_MAX} degrees\n'
    f'  Frequency range: {FMIN}–{FMAX} Hz\n'
    f'--- Results ---\n'
    f'  Events detected: {len(event_idx)}\n'
    f''
)
