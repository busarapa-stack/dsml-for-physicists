"""E3.5  Three physical features from 15 SYNTHETIC force-distance curves, with units,
compared with the values used to generate the curves. Writes data/processed/afm_features.csv.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ".")
from src.features import (adhesion_work_aJ, contact_slope_nN_per_nm, max_adhesion_nN,
                          remove_baseline)

curves = pd.read_csv('data/raw/afm_curves.csv')
truth = pd.read_csv('data/raw/afm_curves_truth.csv', index_col='curve_id')
rows = []
for cid, c in curves.groupby('curve_id'):
    c, base = remove_baseline(c)
    rows.append({'curve_id': cid, 'baseline_nN': base,
                 'slope_nN_per_nm': contact_slope_nN_per_nm(c),
                 'adhesion_nN': max_adhesion_nN(c),
                 'adhesion_nN_sg31': max_adhesion_nN(c, window=31),
                 'work_aJ': adhesion_work_aJ(c)})
feat = pd.DataFrame(rows).set_index('curve_id')
Path('data/processed').mkdir(parents=True, exist_ok=True)
feat.round(4).to_csv('data/processed/afm_features.csv')

for col in ('baseline_nN', 'slope_nN_per_nm', 'adhesion_nN', 'work_aJ'):
    err = feat[col] - truth[col]
    print(f"{col:17s}: mean error {err.mean():+.4f}, rms {np.sqrt((err**2).mean()):.4f}")
err_sg = feat['adhesion_nN_sg31'] - truth['adhesion_nN']
print(f"adhesion after Savitzky-Golay (31): mean error {err_sg.mean():+.3f} nN "
      "-> smoothing biases the peak adhesion low")
extra = 0.5 * truth['adhesion_nN'] * 1.0          # half a 1 nm step at the snap-off
print(f"work: the trapezoid rule joins the last stuck point to the first free point, adding"
      f" about 0.5 x adhesion x 1 nm = {extra.mean():.2f} aJ on average -> sampling limits this feature")
print("units: slope nN/nm = N/m; adhesion nN; work nN*nm = 1e-18 J = aJ")
print("saved data/processed/afm_features.csv")
