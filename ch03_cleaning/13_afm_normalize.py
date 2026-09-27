"""Ch.3  Physics-aware normalisation: turn AFM force back into cantilever deflection.

Book references: ssec:phys-normalize, ex:afm-norm, code:afm-norm, review question 7
The curve is SYNTHETIC. The script also shows the text's point: dividing by the SAME k
that produced the forces gives back the measured deflection exactly, while a different k
scales every point by the ratio of the two values (a systematic, not random, error).
"""
import os
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).resolve().parent)

# --- code:afm-norm ----------------------------------------------------------
import pandas as pd
afm = pd.read_csv('data/raw/afm_spectrum.csv')
k_cantilever = 0.1        # N/m, identical to nN/nm
afm['deflection_nm'] = afm['force_nN'] / k_cantilever

print(afm[['segment', 'distance_nm', 'force_nN', 'deflection_nm']].iloc[[0, 250, 300, 560]]
      .to_string(index=False))

# suppose the instrument measured deflection d and computed F = k_used * d
d_measured = afm['deflection_nm'].values
for k_used in (0.1, 0.12):
    F = k_used * d_measured
    d_back = F / k_cantilever
    print(f"forces made with k = {k_used} N/m, divided by k = {k_cantilever} N/m: "
          f"deflection off by a factor {np.median(d_back / d_measured):.2f} at every point")

# a calibration error in k is systematic: averaging more points does not remove it
rng = np.random.default_rng(3)
d0 = 5.0                                   # nm, true deflection of a repeated measurement
k_true = 0.1                               # N/m
k_cal = 0.11                               # N/m, one calibration that happens to be 10 % high
print("mean force from N repeated measurements (true value "
      f"{k_true * d0:.3f} nN), deflection noise 0.5 nm:")
for n in (10, 1_000, 100_000):
    d = d0 + rng.normal(0, 0.5, n)
    print(f"  N = {n:6d}: {np.mean(k_cal * d):.4f} nN")
print("-> the random part shrinks with N, the 10 % from k stays: a systematic uncertainty")
