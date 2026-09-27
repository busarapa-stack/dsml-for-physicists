"""E3.4  Sensitivity of the mean redshift to the missing-value method, against the truth
(known because the data are SYNTHETIC). Also: what if the photo-z relation is wrong?
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)
df = pd.read_csv('data/raw/galaxy10_metadata.csv', index_col='image_id')
truth = pd.read_csv('data/raw/galaxy10_truth.csv', index_col='image_id')
df = df[df['r_mag'] >= 8]
z_true = truth.loc[df.index, 'z_true']

good = lambda r: 0.05 * (r - 15.0)            # the book's toy relation
wrong = lambda r: 0.04 * (r - 15.0)           # a relation with a 20 % wrong slope
methods = {
    'truth':                      z_true,
    'drop rows':                  df['redshift'].dropna(),
    'fill median':                df['redshift'].fillna(df['redshift'].median()),
    'fill from r_mag (right)':    df['redshift'].fillna(good(df['r_mag'])),
    'fill from r_mag (wrong)':    df['redshift'].fillna(wrong(df['r_mag'])),
}
print(f"{'method':28s} {'mean z':>8s} {'std z':>8s} {'bias of mean':>13s}")
for k, z in methods.items():
    print(f"{k:28s} {z.mean():8.4f} {z.std():8.4f} {z.mean() - z_true.mean():+13.4f}")
print("\nprediction check: dropping rows lowers the mean (faint = far galaxies removed); "
      "median filling also shrinks the spread; the relation-based fill is best only if the "
      "relation is right.")
