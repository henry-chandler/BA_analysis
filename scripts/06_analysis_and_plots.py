# -*- coding: utf-8 -*-
"""
06_analysis_and_plots.py

Final analysis script. Loads quality-annotated infrasound events, weather
station data, and RAMMS::Rockfall trajectory data, then produces a series
of figures comparing rockfall activity with meteorological conditions and
simulated trajectory geometry.

Figures produced:
    1. Weekly event counts — full study period bar chart with snow cover shading
    2. Events vs weather — 4-row subplot (one per year) showing daily event
       counts, air temperature, rainfall, and snow cover periods
    3. Back azimuth vs event duration — observed events (high + medium quality)
       with moving-window mean and percentile bands
    4. Back azimuth vs event duration — high-quality events only
    5. Back azimuth vs trajectory length — RAMMS::Rockfall simulations

Inputs:
    Quality-annotated infrasound events:
        <INF_DIR>/BAEI_quality_events_2022-2026.npy
    Weather station CSV (Barry Arm East Station, 15-min intervals):
        <WEATHER_DIR>/BA_weather_east.csv
    RAMMS trajectory CSV from 04_extract_ramms_trajectories.py:
        <RAMMS_DIR>/BA_rockfall_trajectory_data.csv

Outputs:
    figures/events_weekly.png
    figures/events_weather_subplots.png
    figures/baz_vs_length_observed.png
    figures/baz_vs_length_high_quality.png
    figures/baz_vs_length_ramms.png

Dependencies:
    numpy, pandas, matplotlib, scipy, sklearn
    See requirements.txt for version details.

"""

# ============================================================
# IMPORTS
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import os
from datetime import timedelta, datetime
from scipy.ndimage import gaussian_filter1d

# ============================================================
# USER PARAMETERS — edit these before running
# ============================================================

# Where the figures will be saved
FIGURES_DIR = ''

os.makedirs(FIGURES_DIR, exist_ok=True)

# EVENT_QUAL_FILE is the output file from the script 04_event_reviewer which contains event quality annotations
# WEATHER_FILE is the path and file name of the raw weather data CSV
# RAMMS_TRAJ_FILE is the output file from script 05_extract_RAMMS_trajectory
EVENT_QUAL_FILE   = ''
WEATHER_FILE = ''
RAMMS_TRAJ_FILE = ''

# Quality codes to include in the main analysis dataset (high + medium)
QUALITY_KEEP = ('h', 'm')

# Back azimuth and snow cover filter bounds
# NOTE: these filters were already applied in 02_combine_and_filter_events.py
# and are re-applied here in case the quality-annotated file has not been
# re-filtered (e.g. if quality annotation added events back in).
BAZ_MIN = 250
BAZ_MAX = 305
SNOW_START = (11, 9)   # November 9
SNOW_END = (3, 20)   # March 20

# Winter seasons for shading on time series plots
WINTER_SEASONS = [
    (datetime(2022, 11, 9), datetime(2023, 3, 20)),
    (datetime(2023, 11, 9), datetime(2024, 3, 20)),
    (datetime(2024, 11, 9), datetime(2025, 3, 20)),
    (datetime(2025, 11, 9), datetime(2026, 3, 20)),
]

# ============================================================
# FUNCTIONS
# ============================================================

def snow_cover(date, snow_start=SNOW_START, snow_end=SNOW_END):
    """
    Return True if date falls within the annual snow cover window.

    Parameters
    ----------
    date : datetime-like
        Date to check.
    snow_start : tuple of (int, int)
        (month, day) start of snow cover window.
    snow_end : tuple of (int, int)
        (month, day) end of snow cover window.

    Returns
    -------
    bool
    """
    m, d = date.month, date.day
    start_m, start_d = snow_start
    end_m,   end_d   = snow_end
    after_start = (m > start_m) or (m == start_m and d >= start_d)
    before_end  = (m < end_m)   or (m == end_m   and d <= end_d)
    return after_start or before_end


def is_winter_daily(date):
    """Return True if date falls within the snow cover window (daily version)."""
    return snow_cover(date)


def find_sustained_frost(frost_series, min_days=7):
    """
    Identify periods where frost conditions persist for at least min_days
    consecutive days.

    Parameters
    ----------
    frost_series : pd.Series of int (0 or 1)
        Daily frost indicator.
    min_days : int
        Minimum consecutive days to qualify as sustained frost.

    Returns
    -------
    pd.Series of int (0 or 1)
    """
    values = frost_series.values
    result = np.zeros(len(values))
    i = 0
    while i < len(values):
        if values[i] == 1:
            j = i
            while j < len(values) and values[j] == 1:
                j += 1
            if (j - i) >= min_days:
                result[i:j] = 1
            i = j
        else:
            i += 1
    return pd.Series(result, index=frost_series.index)


def moving_window_stats(x, y, window=15, step=3):
    """
    Compute moving-window statistics of x as a function of y.

    Slides a window of width `window` along the y-axis in steps of `step`,
    computing the mean, 50th, 90th, and 95th percentiles of x values within
    each window.

    Parameters
    ----------
    x : np.ndarray
        Values to compute statistics on (e.g. event duration).
    y : np.ndarray
        Values to bin along (e.g. back azimuth).
    window : float
        Width of the sliding window in y units.
    step : float
        Step size in y units.

    Returns
    -------
    x_mean, x_p50, x_p90, x_p95, y_centers : np.ndarray
    """
    y_centers = np.arange(np.nanmin(y), np.nanmax(y), step)
    x_mean, x_p50, x_p90, x_p95, y_out = [], [], [], [], []
    for yc in y_centers:
        mask = (y >= yc - window / 2) & (y < yc + window / 2)
        if mask.sum() > 3:
            x_mean.append(np.mean(x[mask]))
            x_p50.append(np.percentile(x[mask], 50))
            x_p90.append(np.percentile(x[mask], 90))
            x_p95.append(np.percentile(x[mask], 95))
            y_out.append(yc)
    return (np.array(x_mean), np.array(x_p50), np.array(x_p90),
            np.array(x_p95), np.array(y_out))


def plot_baz_vs_length(lengths, initial_baz, xlabel, save_path=None):
    """
    Scatter plot of back azimuth vs event duration/length with moving-window
    mean and percentile bands.

    Parameters
    ----------
    lengths : np.ndarray
        Event durations or trajectory lengths [s or m].
    initial_baz : np.ndarray
        Back azimuth values [degrees].
    xlabel : str
        X-axis label.
    save_path : str, optional
        Full path to save the figure. If None, figure is shown only.
    """
    x_mean, x_p50, x_p90, x_p95, y_out = moving_window_stats(
        lengths, initial_baz, window=15, step=3)

    fig, ax = plt.subplots()
    ax.scatter(lengths, initial_baz, color='black', s=10, alpha=0.8,
               label='Data')
    ax.plot(x_mean, y_out, color='red', linewidth=2, label='Mean')
    ax.plot(x_p95, y_out, color='#009E73', linewidth=2, linestyle='--',
            label='95th percentile')
    ax.fill_betweenx(y_out, x_p50, x_p90, alpha=0.5, color='#CC79A7',
                     label='50–90th percentile')
    ax.set_xlim([10, 90])
    ax.set_ylim([BAZ_MIN, BAZ_MAX])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Back Azimuth (°)')
    for y in range(260, 310, 10):
        ax.axhline(y, color='grey', linewidth=0.7, alpha=0.7, zorder=0)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()


# ============================================================
# LOAD AND FILTER EVENTS
# ============================================================

print('Loading quality-annotated events...')
df = np.load(EVENT_QUAL_FILE, allow_pickle=True).item()

event_times = np.array(df['event_times'])
event_dur = np.array(df['event_dur'])
initial_baz = np.array(df['initial_baz'])
seismic_rec = np.array(df['seismic_recording'])
event_quality = np.array(df['quality'])

# Filter to high and medium quality events for main analysis
quality_mask = np.isin(event_quality, QUALITY_KEEP)
event_times = event_times[quality_mask]
event_dur = event_dur[quality_mask]
initial_baz = initial_baz[quality_mask]
seismic_rec = seismic_rec[quality_mask]
event_quality = event_quality[quality_mask]

# Keep a high-quality-only subset for separate plots
high_mask = (event_quality == 'h')
event_times_h = event_times[high_mask]
event_dur_h = event_dur[high_mask]
initial_baz_h = initial_baz[high_mask]

event_times_dt = np.array([t.datetime for t in event_times])
times_pd       = pd.to_datetime(event_times_dt)

print(f'Events loaded: {len(event_times)} (after quality filter)')

# ============================================================
# SEISMIC RECORDING STATS
# ============================================================

both_seismo = np.sum(seismic_rec == 'b')
bat_only = np.sum(seismic_rec == 't')
bae_only = np.sum(seismic_rec == 'e')
no_seismo = np.sum(seismic_rec == 'n')

print(f'\nSeismic recording summary ({len(event_times)} events):')
print(f'Both seismometers : {both_seismo}')
print(f'BAT only: {bat_only}')
print(f'BAE only: {bae_only}')
print(f'No seismometer: {no_seismo}')

# ============================================================
# LOAD AND PROCESS WEATHER DATA
# ============================================================

print('\nLoading weather data...')
BA_es = pd.read_csv(WEATHER_FILE)
BA_es['timestamp'] = pd.to_datetime(BA_es['timestamp'])
BA_es['timestamp'] = (BA_es['timestamp']
                      .dt.tz_localize('US/Alaska',
                                      ambiguous=True,
                                      nonexistent='shift_forward')
                      .dt.tz_convert('UTC'))
timestamp = BA_es['timestamp']

# Daily summaries
daily_rain = pd.Series(BA_es['rain_mm_tot'].values,
                             index=pd.to_datetime(timestamp)
                             ).resample('D', label='left', closed='left').sum()
daily_airt_max = pd.Series(BA_es['airt_c_avg'].values,
                             index=pd.to_datetime(timestamp)
                             ).resample('D', label='left', closed='left').max()
daily_airt_min = pd.Series(BA_es['airt_c_avg'].values,
                             index=pd.to_datetime(timestamp)
                             ).resample('D', label='left', closed='left').min()
daily_airt_avg = pd.Series(BA_es['airt_c_avg'].values,
                             index=pd.to_datetime(timestamp)
                             ).resample('D', label='left', closed='left').mean()

# Frost-cycle window: days where avg temp is in the -8 to -3 °C range
frost_window  = ((daily_airt_avg >= -8) & (daily_airt_avg <= -3)).astype(int)
frost_window_7day = frost_window.rolling(window=7).sum()
sustained_frost = find_sustained_frost(frost_window, min_days=7)

# Daily winter mask for shading
timestamp_daily = pd.date_range(
    timestamp.min().floor('D'), timestamp.max().floor('D'), freq='D')
winter_mask_daily = pd.Series(
    [int(is_winter_daily(t)) for t in timestamp_daily],
    index=timestamp_daily)

# Thaw days: max > 0 and min < 0
thaw_mask = (daily_airt_max > 0) & (daily_airt_min < 0)
thaw = np.full_like(daily_airt_max, np.nan, dtype=float)
thaw[thaw_mask] = 1

# Event time series for resampling
ts  = pd.Series(1, index=times_pd)
daily_count = ts.resample('D', label='left', closed='left').sum().fillna(0)
x_daily = daily_count.index

# ============================================================
# FIGURE 1 — Weekly event counts
# ============================================================

print('\nPlotting Figure 1: weekly event counts...')
weekly_counts = ts.resample('7D').sum().fillna(0)
x_weekly      = weekly_counts.index

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x_weekly, weekly_counts, color='grey', width=5,
       label=f'Weekly events ({len(event_times)} total)')
for i, (start, end) in enumerate(WINTER_SEASONS):
    ax.axvspan(start, end, color='steelblue', alpha=0.3,
               label='Snow cover (Nov 9 – Mar 20)' if i == 0 else None)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(True, alpha=0.5)
ax.set_title('Temporal Distribution of Events at Barry Arm')
ax.set_xlabel('Date (UTC)')
ax.set_ylabel('Number of Events')
ax.set_ylim([0, 24])
ax.set_xlim([datetime(2022, 7, 20), datetime(2025, 12, 31)])
plt.tight_layout()
save_path = os.path.join(FIGURES_DIR, 'events_weekly.png')
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Saved: {save_path}')
plt.show()

# ============================================================
# FIGURE 2 — Events vs weather (4-row subplot, one per year)
# ============================================================

print('Plotting Figure 2: events vs weather...')

year_ranges = [
    (pd.Timestamp('2022-09-01'), pd.Timestamp('2023-01-01')),
    (pd.Timestamp('2023-01-01'), pd.Timestamp('2024-01-01')),
    (pd.Timestamp('2024-01-01'), pd.Timestamp('2025-01-01')),
    (pd.Timestamp('2025-01-01'), pd.Timestamp('2026-01-01')),
]
year_labels = ['2022', '2023', '2024', '2025']

fig = plt.figure(figsize=(20, 12))

left_margin = 0.07
right_margin = 0.06
top_margin = 0.08
bottom_margin = 0.08
plot_area_width = 1.0 - left_margin - right_margin
plot_area_height = 1.0 - top_margin - bottom_margin

n_rows  = 4
row_height = plot_area_height / n_rows
row_gap = 0.03
frac_2022 = 4 / 12                            # 2022 data starts in September
row_widths = [frac_2022, 1.0, 1.0, 1.0]

all_lines, all_labels = [], []

for i, ((t1, t2), year_label) in enumerate(zip(year_ranges, year_labels)):

    row_bottom = 1.0 - top_margin - (i + 1) * row_height + row_gap / 2
    ax_width   = plot_area_width * row_widths[i]
    ax_height  = row_height - row_gap

    # 2022 row is right-aligned to share the same right edge as other rows
    ax_left = (left_margin + plot_area_width - ax_width
               if i == 0 else left_margin)

    ax1 = fig.add_axes([ax_left, row_bottom, ax_width, ax_height])
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 55))

    # Events (bar) and frost window (fill)
    ax1.bar(x_daily + timedelta(days=0.5), daily_count, width=1,
            color='black', alpha=0.4, edgecolor='black',
            label='Daily events')
    ax1.fill_between(frost_window_7day.index, frost_window_7day.values,
                     color='orange', alpha=0.2, label='7-day frost window sum')

    # Air temperature
    ax2.plot(timestamp_daily, daily_airt_max, color='red',
             label='Air temp max')
    ax2.plot(timestamp_daily, daily_airt_min, color='blue',
             label='Air temp min')
    ax2.hlines(y=0, color='grey', xmin=t1, xmax=t2, alpha=0.7)

    # Rainfall (inverted axis)
    ax3.bar(timestamp_daily + timedelta(days=0.5), daily_rain, width=1,
            color='royalblue', edgecolor='black', alpha=0.5, label='Rain')
    ax3.invert_yaxis()

    # Winter shading
    in_winter   = False
    first_winter = True
    for j in range(len(winter_mask_daily)):
        val = winter_mask_daily.iloc[j]
        idx = winter_mask_daily.index[j]
        if val == 1 and not in_winter:
            start = idx
            in_winter = True
        elif val == 0 and in_winter:
            ax1.axvspan(start, idx, color='steelblue', alpha=0.2, zorder=0,
                        label='Snow cover period' if first_winter else None)
            in_winter   = False
            first_winter = False
    if in_winter:
        ax1.axvspan(start, winter_mask_daily.index[-1],
                    color='steelblue', alpha=0.15, zorder=0,
                    label='Snow cover period' if first_winter else None)

    ax1.set_xlim(t1, t2)
    ax2.set_xlim(t1, t2)
    ax3.set_xlim(t1, t2)
    ax1.set_ylim(0, 10)
    ax2.set_ylim(-20, 20)
    ax3.set_ylim(150, 0)

    ax2.set_ylabel('Air Temp (°C)', fontsize=15)
    ax3.set_ylabel('Rain (mm)', fontsize=15)
    ax1.set_ylabel(year_label, fontsize=17, rotation=0, labelpad=35,
                   va='center')
    ax1.tick_params(axis='both', labelsize=11)
    ax2.tick_params(axis='y', labelsize=11)
    ax3.tick_params(axis='y', labelsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.tick_params(axis='x', length=6, width=1.2)

    if i < 3:
        ax1.tick_params(labelbottom=False)
    else:
        ax1.set_xlabel('Month', fontsize=15)

    # Collect legend entries from the first full year only
    if i == 0:
        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        l3, lb3 = ax3.get_legend_handles_labels()
        all_lines = l1 + l2 + l3
        all_labels = lb1 + lb2 + lb3

# Legend in the empty space to the left of the 2022 panel
row_bottom_2022 = 1.0 - top_margin - row_height + row_gap / 2
legend_ax = fig.add_axes([
    left_margin,
    row_bottom_2022,
    plot_area_width * (1 - frac_2022) - 0.01,
    row_height - row_gap
])
legend_ax.axis('off')
legend_ax.legend(all_lines, all_labels, loc='center left', ncol=1,
                 frameon=True, fontsize=15, borderaxespad=0,
                 handlelength=2, handleheight=1.2, labelspacing=0.6)

fig.suptitle('Events vs Weather Conditions (2022–2025)',
             y=0.99, fontsize=15, va='top')
fig.text(0.01, 0.5, 'Number of Events\nFCW (7-day sum)',
         va='center', ha='center', rotation='vertical', fontsize=15)

save_path = os.path.join(FIGURES_DIR, 'events_weather_subplots.png')
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Saved: {save_path}')
plt.show()

# ============================================================
# FIGURES 3 & 4 — Back azimuth vs event duration (observed)
# ============================================================

print('Plotting Figures 3 & 4: back azimuth vs event duration...')

plot_baz_vs_length(
    event_dur, initial_baz,
    xlabel='Event Duration (s)',
    save_path=os.path.join(FIGURES_DIR, 'baz_vs_length_observed.png')
)

plot_baz_vs_length(
    event_dur_h, initial_baz_h,
    xlabel='Event Duration — high quality only (s)',
    save_path=os.path.join(FIGURES_DIR, 'baz_vs_length_high_quality.png')
)

# ============================================================
# FIGURE 5 — Back azimuth vs trajectory length (RAMMS)
# ============================================================

print('Plotting Figure 5: back azimuth vs RAMMS trajectory length...')

ramms_df          = pd.read_csv(RAMMS_TRAJ_FILE)
trajectory_length = np.array(ramms_df['trajectory_length'])
ramms_initial_baz = np.array(ramms_df['initial_baz'])

x_mean, x_p50, x_p90, x_p95, y_out = moving_window_stats(
    trajectory_length, ramms_initial_baz, window=15, step=3)

fig, ax = plt.subplots()
ax.scatter(trajectory_length, ramms_initial_baz, edgecolors='none', color='grey',
           s=15, alpha=0.4, label='Data')
ax.plot(x_mean, y_out, color='red', linewidth=3, label='Mean')
ax.plot(x_p95, y_out, color='#009E73', linewidth=3, linestyle='--',
        label='95th percentile')
ax.fill_betweenx(y_out, x_p50, x_p90, alpha=0.5, color='#CC79A7',
                 label='50–90th percentile')
ax.set_xlim([trajectory_length.min() * 0.9, trajectory_length.max() * 1.1])
ax.set_ylim([BAZ_MIN, BAZ_MAX])
ax.set_xlabel('Trajectory Length (m)')
ax.set_ylabel('Back Azimuth (°)')
for y in range(260, 310, 10):
    ax.axhline(y, color='grey', linewidth=0.7, alpha=0.7, zorder=0)
ax.legend()
plt.tight_layout()

save_path = os.path.join(FIGURES_DIR, 'baz_vs_length_ramms.png')
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Saved: {save_path}')
plt.show()

print('\nAll figures complete.')