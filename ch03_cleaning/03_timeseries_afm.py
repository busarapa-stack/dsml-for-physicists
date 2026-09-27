"""Ch.3  AFM force time series: remove spikes with a median filter, then remove drift.
Also: why aliasing cannot be undone afterwards (misconception box, review question 4).

Book references: ssec:timeseries, code:ts-clean
The AFM signal is SYNTHETIC: 2 s at 1 kHz, 1.5 Hz topography, linear drift, 15 spikes.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent)
afm = pd.read_csv('data/raw/afm_timeseries.csv')
spikes = pd.read_csv('data/raw/afm_timeseries_spikes.csv')['spike_index'].values

# --- code:ts-clean ----------------------------------------------------------
from scipy.signal import medfilt, detrend

afm['force_clean'] = medfilt(afm['force_nN'].values, kernel_size=5)
afm['force_detrended'] = detrend(afm['force_clean'].values, type='linear')

# --- checks -----------------------------------------------------------------
raw, clean = afm['force_nN'].values, afm['force_clean'].values
resid = raw - clean
print(f"{len(afm)} samples; {len(spikes)} spikes were injected")
print(f"largest |raw - median-filtered| at the spikes    : {np.abs(resid[spikes]).min():.2f}"
      f" to {np.abs(resid[spikes]).max():.2f} nN")
print(f"typical |raw - median-filtered| elsewhere (95th %): "
      f"{np.percentile(np.abs(np.delete(resid, spikes)), 95):.3f} nN")

# moving average for comparison (review question 3)
mov = pd.Series(raw).rolling(5, center=True, min_periods=1).mean().values
worst = spikes[np.argmax(np.abs(resid[spikes]))]
print(f"at the largest spike: raw {raw[worst]:.2f}, median {clean[worst]:.2f}, "
      f"moving average {mov[worst]:.2f} nN -> the average smears the spike, the median removes it")

slope = np.polyfit(afm['time_s'], afm['force_clean'], 1)[0]
print(f"straight line removed by detrend: slope {slope:.3f} nN/s, but the drift put in was 0.40 nN/s")
print("  -> over only 3 cycles the slow topography also tilts the fitted line; detrend removes"
      " part of the real signal, so plot before/after and state the choice")

fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
ax[0].plot(afm['time_s'], raw, lw=0.5); ax[0].set_ylabel('raw (nN)')
ax[1].plot(afm['time_s'], clean, lw=0.5); ax[1].set_ylabel('median filtered')
ax[2].plot(afm['time_s'], afm['force_detrended'], lw=0.5); ax[2].set_ylabel('detrended')
ax[2].set_xlabel('time (s)')
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('outputs/ch03_afm_timeseries.png', dpi=150)

# --- aliasing: 90 Hz sampled at 100 Hz looks exactly like 10 Hz -------------
fs = 100.0
t = np.arange(0, 1, 1 / fs)
x90 = np.cos(2 * np.pi * 90 * t)
x10 = np.cos(2 * np.pi * 10 * t)
print(f"\naliasing: sampled at {fs:.0f} Hz, a 90 Hz and a 10 Hz cosine differ by at most "
      f"{np.abs(x90 - x10).max():.1e} -> identical samples, no filter can separate them")
print("saved outputs/ch03_afm_timeseries.png")
