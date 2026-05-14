
"""
04_event_reviewer.py

Interactive GUI tool for manual review and quality annotation of detected
infrasound events at Barry Arm, Alaska. Steps through events one-by-one,
displaying waveform and spectrogram panels for the BAEI infrasound array
and any available seismic stations (BAT, BAE), and records per-event
annotations to a persistent .npy file.

For each event the reviewer records:
    seismic_recording : whether seismic data was available (e.g. b = both / t = BAT / e = BAE / n = neither)
    quality           : signal quality assessment (e.g. h = high / m = medium / p = poor)

Annotations are saved incrementally after each event so progress is never
lost if the session is interrupted. Re-running the script automatically
resumes from the first unannotated event.

Inputs:
    Filtered combined event dataset from 03_combine_and_filter_events.py:
        <FILT_EVENT_FILE>
    Raw daily infrasound miniSEED files:
        <RAW_INF_DIR>/BA_YYYY/BAEI..HDF.YYYY-MM-DDTHH-MM-SS_24h_100Hz.ms
    Raw daily seismic miniSEED files:
        <RAW_BAT_DIR>/BAT..BHZ.YYYY-MM-DDTHH-MM-SS_24h_100Hz.ms
        <RAW_BAE_DIR>/BAE..BHZ.YYYY-MM-DDTHH-MM-SS_24h_100Hz.ms

Outputs:
    Annotated event dataset:
        <OUTPUT_FILE>
    Optional high-resolution event figures (300 dpi):
        <FIGURES_DIR>/event_NNNNN.png

Usage:
    Set paths in the USER PARAMETERS section and run:
        python 04_event_reviewer.py
    Use SAVE & NEXT (or press Enter on the Quality field) to annotate
    and advance. Use SKIP to advance without saving. Use JUMP TO EVENT #
    to navigate to a specific event.

Dependencies:
    obspy, numpy, pandas, matplotlib, scipy, tkinter (stdlib)
    See requirements.txt for version details.

"""

# ============================================================
# IMPORTS
# ============================================================

import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.axes_grid1 import make_axes_locatable
from obspy import read
from scipy import signal as sp_signal

# ============================================================
# USER PARAMETERS — edit these before running
# ============================================================

# RAW_INF_DIR: directory containing raw infrasound miniSEED files,
#   organized as BA_YYYY/BAEI..HDF.*.ms
# RAW_BAT_DIR: directory containing raw BAT seismic miniSEED files
# RAW_BAE_DIR: directory containing raw BAE seismic miniSEED files
# FILT_EVENT_FILE: filtered event .npy file from 03_combine_and_filter_events.py
# OUTPUT_FILE: full path (including filename) for the quality-annotated output .npy
# FIGURES_DIR: directory for saving high-resolution event figures

RAW_INF_DIR     = ''
RAW_BAT_DIR     = ''
RAW_BAE_DIR     = ''
FILT_EVENT_FILE = ''
OUTPUT_DIR      = ''
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, 'BAEI_events_quality_2022-2026(FINAL_PRODUCT).npy')
FIGURES_DIR     = ''

# Events to save when "PRINT SELECTED" is clicked — edit this list as needed
BATCH_SAVE_EVENTS = []

# Spectrogram parameters
SPEC_WLEN    = 2
SPEC_OVERLAP = 0.9

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def plot_spectrogram(ax, tr, wlen, overlap, clim_divisor=10, ylim=30):
    """
    Plot a spectrogram on ax using scipy.signal.spectrogram.
    Attaches a small colorbar to the right of the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    tr : obspy.Trace
    wlen : float
        Spectrogram window length [s].
    overlap : float
        Window overlap [proportion].
    clim_divisor : float
        Color limit set to Sxx.max() / clim_divisor.
    ylim : float
        Upper frequency limit [Hz].
    """
    fs       = tr.stats.sampling_rate
    nperseg  = int(wlen * fs)
    noverlap = int(nperseg * overlap)
    f, t, Sxx = sp_signal.spectrogram(tr.data, fs=fs,
                                       nperseg=nperseg, noverlap=noverlap)
    start_num = mdates.date2num(tr.stats.starttime.datetime)
    t_dates   = start_num + t / 86400.0

    pcm = ax.pcolormesh(t_dates, f, Sxx, cmap='plasma', shading='auto',
                        vmin=0, vmax=Sxx.max() / clim_divisor)
    ax.set_ylim(0, ylim)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb  = ax.get_figure().colorbar(pcm, cax=cax)
    cb.ax.tick_params(labelsize=6)
    cb.set_label('PSD', fontsize=6, labelpad=2)


def build_figure(i, event_times, event_dur, raw_inf_dir,
                 raw_bat_dir, raw_bae_dir):
    """
    Build and return a matplotlib Figure for event i, showing waveform and
    spectrogram panels for infrasound (BAEI) and any available seismic
    stations (BAT, BAE). Each panel is labelled with its station name.

    Parameters
    ----------
    i : int
        Event index.
    event_times : np.ndarray of UTCDateTime
        Event start times.
    event_dur : np.ndarray
        Event durations [s].
    raw_inf_dir : str
        Directory containing raw infrasound miniSEED files, organized as
        BA_YYYY/BAEI..HDF.*.ms
    raw_bat_dir : str
        Directory containing raw BAT seismic miniSEED files.
    raw_bae_dir : str
        Directory containing raw BAE seismic miniSEED files.

    Returns
    -------
    matplotlib.figure.Figure
    """
    date_str = (f'{event_times[i].year:04d}-{event_times[i].month:02d}-'
                f'{event_times[i].day:02d}T00-00-00_24h_100Hz.ms')

    inf_file = os.path.join(raw_inf_dir, f'BAEI..HDF.{date_str}')
    bat_file = os.path.join(raw_bat_dir, f'BAT..BHZ.{date_str}')
    bae_file = os.path.join(raw_bae_dir, f'BAE..BHZ.{date_str}')

    # Each entry: (display label, filepath, y-axis label, spectrogram freq limit)
    source_defs = [
        ('BAEI — Infrasound (HDF)', inf_file,  'Pressure (Pa)',   30),
    ]
    if os.path.exists(bat_file):
        source_defs.append(('BAT — Seismic (BHZ)', bat_file, 'Velocity (m/s)', 20))
    if os.path.exists(bae_file):
        source_defs.append(('BAE — Seismic (BHZ)', bae_file, 'Velocity (m/s)', 20))

    plot_start = event_times[i] - 30
    plot_end   = event_times[i] + event_dur[i] + 30

    # Load and preprocess each source
    streams        = {}
    active_sources = []
    for label, fpath, ylabel, spec_ylim in source_defs:
        try:
            st = read(fpath).copy().trim(plot_start, plot_end)
            st.detrend('demean')
            st.taper(0.05)
            st.filter('bandpass', freqmin=1.0, freqmax=30.0,
                      corners=4, zerophase=True)
            streams[label] = st[0]
            active_sources.append((label, fpath, ylabel, spec_ylim))
        except Exception as e:
            print(f'  Could not load {label}: {e}')

    # No data available for this event
    if not streams:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, f'Event {i}: No data available',
                ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
        return fig

    n_rows     = len(active_sources)
    inf_onset  = mdates.date2num(event_times[i].datetime)
    inf_offset = mdates.date2num((event_times[i] + event_dur[i]).datetime)

    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(14, 2.2 * n_rows),
                             gridspec_kw={'hspace': 0.20, 'wspace': 0.25})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.text(0.01, 0.99, f'Event {i}', fontsize=11, fontweight='bold',
             va='top', ha='left', transform=fig.transFigure)

    for row, (label, _, ylabel, spec_ylim) in enumerate(active_sources):
        if label not in streams:
            continue
        tr      = streams[label]
        ax_raw  = axes[row, 0]
        ax_spec = axes[row, 1]

        # Waveform
        ax_raw.plot(tr.times('matplotlib'), tr.data, 'k', linewidth=0.6)
        ax_raw.set_ylabel(ylabel, fontsize=9)
        ax_raw.xaxis_date()
        ax_raw.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        # Station label — top-left corner of waveform panel
        ax_raw.text(0.01, 0.97, label,
                    transform=ax_raw.transAxes,
                    fontsize=8, fontweight='bold',
                    va='top', ha='left', color='black',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white', alpha=0.7, ec='none'))

        # Infrasound onset/offset markers on the top row only
        if row == 0:
            ax_raw.axvline(inf_onset,  color='red',  linestyle='--',
                           linewidth=1.5, label='Infrasound onset')
            ax_raw.axvline(inf_offset, color='blue', linestyle='--',
                           linewidth=1.5, label='Infrasound offset')
            ax_raw.legend(fontsize=7, loc='upper right')

        # Spectrogram
        plot_spectrogram(ax_spec, tr, SPEC_WLEN, SPEC_OVERLAP,
                         ylim=spec_ylim)
        ax_spec.set_ylabel('Frequency (Hz)', fontsize=9)

        ax_spec.xaxis.set_major_locator(mdates.SecondLocator(bysecond=[0, 15, 30, 45]))
        ax_spec.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax_spec.tick_params(axis='x', which='minor', length=4)

        # Station label — top-left corner of spectrogram panel
        ax_spec.text(0.01, 0.97, label,
                     transform=ax_spec.transAxes,
                     fontsize=8, fontweight='bold',
                     va='top', ha='left', color='white',
                     bbox=dict(boxstyle='round,pad=0.2',
                               fc='black', alpha=0.5, ec='none'))

        if row == n_rows - 1:
            for ax in [ax_raw, ax_spec]:
                ax.set_xlabel('UTC Time', fontsize=9)
                plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
        else:
            for ax in [ax_raw, ax_spec]:
                plt.setp(ax.get_xticklabels(), visible=False)

    fig.suptitle(f'Infrasound & Seismic Data\n{plot_start} → {plot_end}',
                 fontsize=12)
    return fig


# ============================================================
# MAIN APPLICATION
# ============================================================

class EventReviewer:
    def __init__(self, root):
        self.root = root
        self.root.title('Event Reviewer')

        # Load event data
        df               = np.load(FILT_EVENT_FILE, allow_pickle=True).item()
        self.event_times = np.array(df['event_times'])
        self.event_dur   = np.array(df['event_dur'])
        self.initial_baz = np.array(df['initial_baz'])
        self.n_events    = len(self.event_times)

        # Load or create annotation table
        if os.path.exists(OUTPUT_FILE):
            saved = np.load(OUTPUT_FILE, allow_pickle=True).item()
            self.annotations = pd.DataFrame(saved).fillna('')
            if 'event_number' not in self.annotations.columns:
                self.annotations.index.name = 'event_number'
            else:
                self.annotations = self.annotations.set_index('event_number')
            existing_idx = set(self.annotations.index)
            new_rows = [{'event_number':      i,
                         'seismic_recording': '',
                         'quality':           ''}
                        for i in range(self.n_events) if i not in existing_idx]
            if new_rows:
                self.annotations = pd.concat(
                    [self.annotations,
                     pd.DataFrame(new_rows).set_index('event_number')],
                    axis=0).sort_index()
        else:
            self.annotations = pd.DataFrame({
                'event_number':      range(self.n_events),
                'seismic_recording': [''] * self.n_events,
                'quality':           [''] * self.n_events,
            }).set_index('event_number')

        # Resume from first unannotated event
        unannotated = self.annotations[
            (self.annotations['seismic_recording'] == '') |
            (self.annotations['seismic_recording'].isna())
        ].index
        self.current_idx = int(unannotated[0]) if len(unannotated) > 0 else 0

        self._build_ui()
        self._load_event(self.current_idx)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        DARK   = '#1a1a2e'
        PANEL  = '#16213e'
        CARD   = '#0f3460'
        ACCENT = '#e94560'
        TEXT   = '#eaeaea'
        MUTED  = '#8892a4'

        self.root.configure(bg=DARK)

        # Top bar
        top_bar = tk.Frame(self.root, bg=PANEL, height=50)
        top_bar.pack(fill=tk.X, side=tk.TOP)
        tk.Label(top_bar, text='⬡ EVENT REVIEWER',
                 font=('Courier New', 14, 'bold'),
                 bg=PANEL, fg=ACCENT).pack(side=tk.LEFT, padx=16, pady=10)
        self.progress_label = tk.Label(top_bar, text='',
                                       font=('Courier New', 10),
                                       bg=PANEL, fg=MUTED)
        self.progress_label.pack(side=tk.RIGHT, padx=16)

        # Main pane: plot (left) + controls (right)
        main_pane = tk.Frame(self.root, bg=DARK)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        main_pane.columnconfigure(0, weight=1)
        main_pane.columnconfigure(1, weight=0)
        main_pane.rowconfigure(0, weight=1)

        self.plot_frame = tk.Frame(main_pane, bg=DARK)
        self.plot_frame.grid(row=0, column=0, sticky='nsew')

        right = tk.Frame(main_pane, bg=PANEL, width=300)
        right.grid(row=0, column=1, sticky='nsew', padx=(10, 0))
        right.grid_propagate(False)

        # Event info card
        info_card = tk.Frame(right, bg=CARD, padx=12, pady=10)
        info_card.pack(fill=tk.X, padx=10, pady=(14, 6))
        tk.Label(info_card, text='EVENT INFO',
                 font=('Courier New', 9, 'bold'),
                 bg=CARD, fg=ACCENT).pack(anchor='w')
        self.info_event = tk.Label(info_card, text='',
                                   font=('Courier New', 11, 'bold'),
                                   bg=CARD, fg=TEXT)
        self.info_event.pack(anchor='w', pady=(4, 0))
        self.info_time = tk.Label(info_card, text='',
                                  font=('Courier New', 8),
                                  bg=CARD, fg=MUTED,
                                  wraplength=240, justify='left')
        self.info_time.pack(anchor='w')

        # Annotation form
        form = tk.Frame(right, bg=PANEL, padx=10)
        form.pack(fill=tk.X, pady=6)

        def field(parent, label_text, hint=''):
            tk.Label(parent, text=label_text,
                     font=('Courier New', 9, 'bold'),
                     bg=PANEL, fg=ACCENT).pack(anchor='w', pady=(10, 0))
            if hint:
                tk.Label(parent, text=hint,
                         font=('Courier New', 7),
                         bg=PANEL, fg=MUTED).pack(anchor='w')
            e = tk.Entry(parent, font=('Courier New', 10),
                         bg=CARD, fg=TEXT, insertbackground=TEXT,
                         relief=tk.FLAT, highlightthickness=1,
                         highlightbackground=MUTED, highlightcolor=ACCENT)
            e.pack(fill=tk.X, pady=(2, 0), ipady=5)
            return e

        self.entry_seismic_recording = field(
            form, 'SEISMIC RECORDING',
            'b = both / t = BAT only / e = BAE only / n = neither')
        self.entry_quality = field(
            form, 'QUALITY',
            'h = high / m = medium / p = poor')
        self.entry_quality.bind('<Return>', lambda e: self._save_and_next())

        # Navigation buttons
        btn_frame = tk.Frame(right, bg=PANEL, padx=10)
        btn_frame.pack(fill=tk.X, pady=10)

        def styled_btn(parent, text, cmd, color=ACCENT, side=tk.LEFT):
            b = tk.Button(parent, text=text, command=cmd,
                          font=('Courier New', 9, 'bold'),
                          bg=color, fg='#ffffff',
                          activebackground='#c73652',
                          activeforeground='#fff',
                          relief=tk.FLAT, padx=8, pady=6, cursor='hand2')
            b.pack(side=side, fill=tk.X, expand=True, padx=3)
            return b

        styled_btn(btn_frame, '◀  PREV',        self._prev_event,
                   color='#334155')
        styled_btn(btn_frame, 'SAVE & NEXT  ▶', self._save_and_next,
                   color=ACCENT)

        skip_frame = tk.Frame(right, bg=PANEL, padx=10)
        skip_frame.pack(fill=tk.X)
        styled_btn(skip_frame, 'SKIP (no save)', self._skip_event,
                   color='#334155', side=tk.LEFT)

        # Jump-to field
        jump_frame = tk.Frame(right, bg=PANEL, padx=10, pady=8)
        jump_frame.pack(fill=tk.X)
        tk.Label(jump_frame, text='JUMP TO EVENT #',
                 font=('Courier New', 8), bg=PANEL, fg=MUTED).pack(anchor='w')
        jf = tk.Frame(jump_frame, bg=PANEL)
        jf.pack(fill=tk.X)
        self.jump_entry = tk.Entry(jf, font=('Courier New', 10),
                                   bg=CARD, fg=TEXT, insertbackground=TEXT,
                                   relief=tk.FLAT, highlightthickness=1,
                                   highlightbackground=MUTED,
                                   highlightcolor=ACCENT, width=8)
        self.jump_entry.pack(side=tk.LEFT, ipady=4, padx=(0, 4))
        tk.Button(jf, text='GO', command=self._jump_to_event,
                  font=('Courier New', 9, 'bold'), bg=CARD, fg=ACCENT,
                  relief=tk.FLAT, padx=6, pady=4,
                  cursor='hand2').pack(side=tk.LEFT)

        # Save single figure
        save_fig_frame = tk.Frame(right, bg=PANEL, padx=10, pady=4)
        save_fig_frame.pack(fill=tk.X)
        tk.Label(save_fig_frame, text='SAVE HIGH-RES FIGURE',
                 font=('Courier New', 8, 'bold'),
                 bg=PANEL, fg=ACCENT).pack(anchor='w')
        tk.Label(save_fig_frame, text='event # to save (blank = current)',
                 font=('Courier New', 7), bg=PANEL, fg=MUTED).pack(anchor='w')
        sf = tk.Frame(save_fig_frame, bg=PANEL)
        sf.pack(fill=tk.X, pady=(2, 0))
        self.save_fig_entry = tk.Entry(sf, font=('Courier New', 10),
                                       bg=CARD, fg=TEXT,
                                       insertbackground=TEXT,
                                       relief=tk.FLAT, highlightthickness=1,
                                       highlightbackground=MUTED,
                                       highlightcolor=ACCENT, width=8)
        self.save_fig_entry.pack(side=tk.LEFT, ipady=4, padx=(0, 4))
        tk.Button(sf, text='SAVE PNG', command=self._save_figure_hires,
                  font=('Courier New', 9, 'bold'), bg=CARD, fg=ACCENT,
                  relief=tk.FLAT, padx=6, pady=4,
                  cursor='hand2').pack(side=tk.LEFT)

        # Batch-save selected events
        batch_frame = tk.Frame(right, bg=PANEL, padx=10, pady=4)
        batch_frame.pack(fill=tk.X)
        batch_hint = ', '.join(str(n) for n in BATCH_SAVE_EVENTS) or 'none set'
        tk.Label(batch_frame, text='PRINT SELECTED EVENTS',
                 font=('Courier New', 8, 'bold'),
                 bg=PANEL, fg=ACCENT).pack(anchor='w')
        tk.Label(batch_frame, text=f'events: {batch_hint}',
                 font=('Courier New', 7), bg=PANEL, fg=MUTED,
                 wraplength=260, justify='left').pack(anchor='w')
        tk.Button(batch_frame, text='PRINT SELECTED',
                  command=self._batch_save_figures,
                  font=('Courier New', 9, 'bold'), bg=CARD, fg=ACCENT,
                  relief=tk.FLAT, padx=6, pady=4,
                  cursor='hand2').pack(anchor='w', pady=(3, 0))

        # Recent annotations table
        tk.Label(right, text='RECENT ANNOTATIONS',
                 font=('Courier New', 8, 'bold'),
                 bg=PANEL, fg=MUTED).pack(anchor='w', padx=10, pady=(8, 2))
        table_frame = tk.Frame(right, bg=PANEL, padx=10)
        table_frame.pack(fill=tk.BOTH, expand=True)

        cols = ('Event', 'Recording', 'Quality')
        self.tree = ttk.Treeview(table_frame, columns=cols,
                                 show='headings', height=8)
        style = ttk.Style()
        style.theme_use('default')
        style.configure('Treeview',
                        background=CARD, foreground=TEXT,
                        fieldbackground=CARD, rowheight=20,
                        font=('Courier New', 8))
        style.configure('Treeview.Heading',
                        background=PANEL, foreground=ACCENT,
                        font=('Courier New', 8, 'bold'))
        style.map('Treeview', background=[('selected', ACCENT)])
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=72, anchor='center')
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value='Ready.')
        tk.Label(self.root, textvariable=self.status_var,
                 font=('Courier New', 8), bg=PANEL, fg=MUTED,
                 anchor='w', padx=10).pack(fill=tk.X, side=tk.BOTTOM)

    # ── Event Loading ─────────────────────────────────────────────────────────

    def _load_event(self, idx):
        self.current_idx = idx
        completed = int((self.annotations[['seismic_recording', 'quality']]
                         .ne('').all(axis=1)).sum())
        self.progress_label.config(
            text=f'Event {idx} / {self.n_events - 1}   |   '
                 f'{completed}/{self.n_events} annotated')
        self.info_event.config(text=f'Event #{idx}')
        self.info_time.config(text=str(self.event_times[idx]))
        self.status_var.set(f'Loading event {idx}…')
        self.root.update()

        fig = build_figure(idx, self.event_times, self.event_dur,
                           RAW_INF_DIR, RAW_BAT_DIR, RAW_BAE_DIR)

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        plt.close(fig)

        row = self.annotations.loc[idx]
        for entry, col in [(self.entry_seismic_recording, 'seismic_recording'),
                           (self.entry_quality,           'quality')]:
            entry.delete(0, tk.END)
            val = row[col]
            if pd.notna(val) and str(val) != 'nan':
                entry.insert(0, str(val))

        self.entry_seismic_recording.focus_set()
        self._refresh_table()
        self.status_var.set(f'Event {idx} loaded.')

    # ── Save Logic ────────────────────────────────────────────────────────────

    def _save_current(self):
        idx = self.current_idx
        self.annotations.loc[idx, 'seismic_recording'] = \
            self.entry_seismic_recording.get().strip()
        self.annotations.loc[idx, 'quality'] = \
            self.entry_quality.get().strip()
        self._save_all_outputs()
        self.status_var.set(f'Saved event {idx} → {OUTPUT_FILE}')
        self._refresh_table()

    def _save_all_outputs(self):
        full = pd.DataFrame({
            'event_number': range(self.n_events),
            'event_times':  self.event_times,
            'event_dur':    self.event_dur,
            'initial_baz':  self.initial_baz,
        }).set_index('event_number')
        full['seismic_recording'] = self.annotations['seismic_recording']
        full['quality']           = self.annotations['quality']
        full = full.fillna('')

        def df_to_dict(d):
            out = {'event_number': d.index.values}
            out.update({col: d[col].values for col in d.columns})
            return out

        np.save(OUTPUT_FILE, df_to_dict(full))

    def _save_and_next(self):
        self._save_current()
        if self.current_idx < self.n_events - 1:
            self._load_event(self.current_idx + 1)
        else:
            messagebox.showinfo('Done!', 'All events have been reviewed! 🎉')

    def _prev_event(self):
        if self.current_idx > 0:
            self._load_event(self.current_idx - 1)

    def _skip_event(self):
        if self.current_idx < self.n_events - 1:
            self._load_event(self.current_idx + 1)

    def _jump_to_event(self):
        try:
            target = int(self.jump_entry.get().strip())
            if 0 <= target < self.n_events:
                self._load_event(target)
            else:
                messagebox.showerror(
                    'Out of range',
                    f'Enter a number between 0 and {self.n_events - 1}.')
        except ValueError:
            messagebox.showerror('Invalid input',
                                 'Please enter an integer event number.')

    # ── High-res Figure Save (single) ─────────────────────────────────────────

    def _save_figure_hires(self):
        """Save a 300-dpi PNG for the requested event number."""
        raw = self.save_fig_entry.get().strip()
        if raw == '':
            idx = self.current_idx
        else:
            try:
                idx = int(raw)
            except ValueError:
                messagebox.showerror('Invalid input',
                                     'Please enter an integer event number.')
                return
            if not (0 <= idx < self.n_events):
                messagebox.showerror(
                    'Out of range',
                    f'Enter a number between 0 and {self.n_events - 1}.')
                return
        self._render_and_save(idx)

    # ── Batch Save Selected Events ────────────────────────────────────────────

    def _batch_save_figures(self):
        """Save high-res figures for every event in BATCH_SAVE_EVENTS."""
        if not BATCH_SAVE_EVENTS:
            messagebox.showinfo('Nothing to save',
                                'BATCH_SAVE_EVENTS is empty. '
                                'Add event numbers to the list in USER PARAMETERS.')
            return

        invalid = [n for n in BATCH_SAVE_EVENTS
                   if not (0 <= n < self.n_events)]
        if invalid:
            messagebox.showerror(
                'Out of range',
                f'The following event numbers are out of range '
                f'(0–{self.n_events - 1}) and will be skipped:\n{invalid}')

        to_save = [n for n in BATCH_SAVE_EVENTS if 0 <= n < self.n_events]
        if not to_save:
            return

        saved, failed = [], []
        for idx in to_save:
            self.status_var.set(
                f'Rendering event {idx} '
                f'({to_save.index(idx)+1}/{len(to_save)})…')
            self.root.update()
            try:
                out_path = self._render_and_save(idx, silent=True)
                saved.append((idx, out_path))
            except Exception as e:
                failed.append((idx, str(e)))

        msg_lines = [f'Saved {len(saved)} of {len(to_save)} figures '
                     f'to:\n{FIGURES_DIR}\n']
        if saved:
            msg_lines.append('Saved:  ' + ', '.join(str(n) for n, _ in saved))
        if failed:
            msg_lines.append('\nFailed:')
            for n, err in failed:
                msg_lines.append(f'  Event {n}: {err}')
        messagebox.showinfo('Batch save complete', '\n'.join(msg_lines))
        self.status_var.set(f'Batch save done — {len(saved)} figures written.')

    # ── Shared render-and-save helper ─────────────────────────────────────────

    def _render_and_save(self, idx, silent=False):
        """Render event idx and write a 300-dpi PNG. Returns the output path."""
        self.status_var.set(f'Rendering high-res figure for event {idx}…')
        self.root.update()
        try:
            os.makedirs(FIGURES_DIR, exist_ok=True)
            fig = build_figure(idx, self.event_times, self.event_dur,
                               RAW_INF_DIR, RAW_BAT_DIR, RAW_BAE_DIR)
            out_path = os.path.join(FIGURES_DIR, f'event_{idx:05d}.png')
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.status_var.set(f'Saved → {out_path}')
            if not silent:
                messagebox.showinfo('Saved', f'Figure saved to:\n{out_path}')
            return out_path
        except Exception as e:
            plt.close('all')
            if not silent:
                messagebox.showerror('Save failed', str(e))
                self.status_var.set('Save failed — see error dialog.')
            raise

    # ── Table Refresh ─────────────────────────────────────────────────────────

    def _refresh_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.annotations[['seismic_recording', 'quality']] = (
            self.annotations[['seismic_recording', 'quality']]
            .fillna('').astype(str)
        )
        recent = self.annotations[
            self.annotations['seismic_recording'].ne('')].tail(30)
        for i, r in recent.iterrows():
            self.tree.insert('', tk.END, values=(
                i,
                r['seismic_recording'][:8],
                r['quality'][:8],
            ))
        if self.tree.get_children():
            self.tree.yview_moveto(1.0)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('1400x860')
    root.minsize(1000, 600)
    app = EventReviewer(root)
    root.mainloop()