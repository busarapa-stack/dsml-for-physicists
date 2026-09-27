"""Ch.3  Force-distance curve: remove the baseline, smooth with Savitzky-Golay, and check
that smoothing does not destroy the snap-off (adhesion) event.

Book references: ssec:spectrum, code:spectrum-clean
The curve is SYNTHETIC: baseline offset 0.8 nN, contact slope 0.25 nN/nm, adhesion 3.0 nN.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent)
afm_spec = pd.read_csv('data/raw/afm_spectrum.csv')

# --- code:spectrum-clean ----------------------------------------------------
from scipy.signal import savgol_filter

# baseline: mean force where the tip is far from the surface
far_mask = afm_spec['distance_nm'] > 200
baseline = afm_spec.loc[far_mask, 'force_nN'].mean()
afm_spec['force_corrected'] = afm_spec['force_nN'] - baseline

afm_spec['force_smooth'] = savgol_filter(
    afm_spec['force_corrected'].values,
    window_length=11, polyorder=3
)

# --- checks -----------------------------------------------------------------
print(f"baseline from distance > 200 nm: {baseline:.3f} nN (0.8 nN was put in)")
ret = afm_spec['segment'] == 'retract'
print(f"adhesion (minimum of retract curve), true value 3.0 nN:")
print(f"  corrected, not smoothed       : {-afm_spec.loc[ret, 'force_corrected'].min():.2f} nN")
print(f"  Savitzky-Golay, window 11     : {-afm_spec.loc[ret, 'force_smooth'].min():.2f} nN")
for w in (11, 31, 61):
    ma = afm_spec.loc[ret, 'force_corrected'].rolling(w, center=True, min_periods=1).mean()
    sg = savgol_filter(afm_spec.loc[ret, 'force_corrected'].values, w, 3)
    print(f"  window {w:2d}: moving average {-ma.min():.2f} nN, Savitzky-Golay {-sg.min():.2f} nN")
print("-> wide windows flatten the sudden snap-off; Savitzky-Golay keeps it better"
      " than a moving average of the same width")

fig, ax = plt.subplots(figsize=(7, 4))
r = afm_spec[ret]
ax.plot(r['distance_nm'], r['force_corrected'], '.', ms=2, label='corrected')
ax.plot(r['distance_nm'], r['force_smooth'], lw=1.2, label='Savitzky-Golay (11, 3)')
ax.plot(r['distance_nm'], savgol_filter(r['force_corrected'].values, 61, 3), lw=1.2,
        label='Savitzky-Golay (61, 3): too wide')
ax.set_xlim(0, 80); ax.set_xlabel('distance (nm)'); ax.set_ylabel('force (nN)')
ax.set_title('retract curve: before and after smoothing'); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('outputs/ch03_afm_smoothing.png', dpi=150)
print("saved outputs/ch03_afm_smoothing.png")
