"""E3.2  Clean the messy Galaxy10-style table (SYNTHETIC, ~500 rows).

Writes data/processed/galaxy10_cleaned.csv and prints every decision with its count.
Rule of the book: MARK problems with flag columns, do not delete rows.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)

# 1. look before converting: the sentinel is invisible to isna()
raw = pd.read_csv('data/raw/galaxy10_messy.csv')
print("isna() before handling the sentinel:", raw.isna().sum().to_dict())
print(raw.describe().loc[['min', 'mean', 'max']].round(3))
print(f"mean redshift with -999 left in: {raw['redshift'].mean():.2f}  <- negative, meaningless\n")

# 2. load again, telling pandas that -999 means 'not measured'
df = pd.read_csv('data/raw/galaxy10_messy.csv', na_values=[-999])
n_sent = (raw[['redshift', 'r_mag']] == -999).sum()
print(f"sentinels turned into NaN: redshift {n_sent['redshift']}, r_mag {n_sent['r_mag']}")

# 3. flags instead of deletion
df['flag_no_redshift'] = df['redshift'].isna()
df['flag_negative_z']  = df['redshift'] < 0            # physically impossible here
df['flag_r_range']     = df['r_mag'].notna() & ~df['r_mag'].between(8, 25)
df['flag_no_r_mag']    = df['r_mag'].isna()
df['usable'] = ~(df[['flag_no_redshift', 'flag_negative_z', 'flag_r_range',
                     'flag_no_r_mag']].any(axis=1))
for c in [c for c in df.columns if c.startswith('flag_')]:
    print(f"  {c:18s}: {df[c].sum():3d} rows")
print(f"  usable for redshift work: {df['usable'].sum()} of {len(df)}")

out = Path('data/processed'); out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / 'galaxy10_cleaned.csv', index=False)
print(f"\nsaved data/processed/galaxy10_cleaned.csv ({len(df)} rows, nothing deleted)")
print(f"mean redshift of usable rows: {df.loc[df['usable'], 'redshift'].mean():.4f}")
