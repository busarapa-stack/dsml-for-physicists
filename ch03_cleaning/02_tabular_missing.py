"""Ch.3  Count missing values, mark instead of delete, and ask WHY values are missing.

Book references: ssec:tabular, code:tabular-missing, code:tabular-mask
Text: "suppose redshift is missing for about 15 %" -> in this synthetic table 15.3 %.
The table is SYNTHETIC (made by 01_make_raw_data.py); redshifts are missing more often
for faint galaxies, i.e. not completely at random.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)      # so the book's relative paths work

# --- code:tabular-missing ---------------------------------------------------
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/galaxy10_metadata.csv')
print(df.shape)              # (rows, columns)
print(df.isna().sum())       # number of NaN in each column

# --- code:tabular-mask ------------------------------------------------------
df['has_redshift'] = df['redshift'].notna()
print(f"with redshift: {df['has_redshift'].sum()} / {len(df)}")
print(f"missing fraction: {100 * (1 - df['has_redshift'].mean()):.1f} %")

# --- is the missingness random? compare brightness of the two groups --------
r = df['r_mag']
print("\nmedian r_mag with redshift   :", round(r[df['has_redshift']].median(), 2))
print("median r_mag without redshift:", round(r[~df['has_redshift']].median(), 2))
bins = pd.cut(r, [0, 8, 16, 17, 18, 19, 20, 30])
print("\nmissing fraction by r_mag bin (larger r_mag = fainter):")
print((1 - df.groupby(bins, observed=True)['has_redshift'].mean()).round(3).to_string())
print("-> missing values concentrate in faint galaxies: NOT missing completely at random,"
      " so dropping those rows biases the sample toward bright galaxies")

# --- the r < 8 check from the text ------------------------------------------
too_bright = df[df['r_mag'] < 8]
print(f"\nobjects with r_mag < 8 (probably foreground stars): {len(too_bright)}")
print(too_bright[['image_id', 'r_mag', 'redshift']].to_string(index=False))
