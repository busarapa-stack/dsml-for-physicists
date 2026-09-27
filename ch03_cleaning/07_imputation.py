"""Ch.3  Three ways to fill missing redshifts and what each one does to the mean and spread.

Book references: ssec:missing, tab:imputation, code:imputation, misconception box on mean imputation
Because the data are SYNTHETIC, the true redshift of every galaxy is known
(data/raw/galaxy10_truth.csv), so each method can be compared with the truth.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent)
df = pd.read_csv('data/raw/galaxy10_metadata.csv', index_col='image_id')
truth = pd.read_csv('data/raw/galaxy10_truth.csv', index_col='image_id')
df = df[df['r_mag'] >= 8]                       # drop the 5 foreground stars first
truth = truth.loc[df.index]

# --- code:imputation --------------------------------------------------------
df['z_imp_mean']   = df['redshift'].fillna(df['redshift'].mean())
df['z_imp_median'] = df['redshift'].fillna(df['redshift'].median())

# toy photometric-redshift relation (for teaching only)
def photo_z_estimate(r_mag):
    return 0.05 * (r_mag - 15.0)

df['z_imp_photo'] = df['redshift'].fillna(photo_z_estimate(df['r_mag']))

# --- compare ----------------------------------------------------------------
miss = df['redshift'].isna()
rows = {
    'truth (all galaxies)':   truth['z_true'],
    'drop missing rows':      df['redshift'].dropna(),
    'fill with mean':         df['z_imp_mean'],
    'fill with median':       df['z_imp_median'],
    'fill from r_mag':        df['z_imp_photo'],
}
print(f"{len(df)} galaxies, {miss.sum()} missing redshifts ({100 * miss.mean():.1f} %)\n")
print(f"{'method':24s} {'N':>5s} {'mean z':>8s} {'std z':>8s}")
for name, z in rows.items():
    print(f"{name:24s} {z.size:5d} {z.mean():8.4f} {z.std():8.4f}")
err = (df.loc[miss, 'z_imp_photo'] - truth.loc[miss, 'z_true'])
print(f"\nfilled values only: mean-fill error {np.mean(df.loc[miss, 'z_imp_mean'] - truth.loc[miss, 'z_true']):+.3f}, "
      f"photo-z error {err.mean():+.3f} +- {err.std():.3f}")
print("in this synthetic world the toy relation is right by construction; with a wrong"
      " relation its error would enter every filled value")
