"""Ch.3  pandas basics on the (SYNTHETIC) Galaxy10-style table: load, check dtypes,
select, filter, group.

Book references: sec:pandas, code:df-load, code:dtype-check, code:df-select,
                 code:df-filter, code:groupby
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

# --- code:df-load -----------------------------------------------------------
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/galaxy10_metadata.csv', index_col='image_id')

print(df.shape)
print(df.dtypes)             # check the type of every column
print(df.head())

# --- code:dtype-check -------------------------------------------------------
expected_numeric = ['redshift', 'r_mag', 'g_mag']
for col in expected_numeric:
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"{col} dtype = {df[col].dtype} (expected numeric)")

# why? a few entries are text
bad = pd.to_numeric(df['g_mag'], errors='coerce').isna() & df['g_mag'].notna()
print(f"g_mag entries that are not numbers: {df.loc[bad, 'g_mag'].unique().tolist()} "
      f"in {bad.sum()} rows -> convert with pd.to_numeric(errors='coerce')")
df['g_mag'] = pd.to_numeric(df['g_mag'], errors='coerce')

# --- code:df-select ---------------------------------------------------------
# one column -> Series
redshifts = df['redshift']

# several columns -> DataFrame
photometry = df[['r_mag', 'g_mag']]

# rows by label (loc) or by position (iloc)
row_galaxy_42 = df.loc['galaxy_00042']
row_first_10  = df.iloc[:10]
print(f"\ntypes: {type(redshifts).__name__}, {type(photometry).__name__}, "
      f"{type(row_galaxy_42).__name__}, {type(row_first_10).__name__} {row_first_10.shape}")
print(row_galaxy_42.to_dict())

# --- code:df-filter ---------------------------------------------------------
z_min, z_max = 0.05, 0.15
mask = df['redshift'].notna() & df['redshift'].between(z_min, z_max)
low_z = df[mask]
print(f"\n{len(low_z)} galaxies with {z_min} <= z <= {z_max}")
print(f"between() alone already drops NaN: {df['redshift'].between(z_min, z_max).sum()} "
      "-> the notna() makes the intention explicit")

# --- code:groupby -----------------------------------------------------------
summary = df.groupby('galaxy_class').agg(
    count      = ('redshift', 'size'),
    n_z        = ('redshift', 'count'),     # non-missing only
    mean_z     = ('redshift', 'mean'),
    median_z   = ('redshift', 'median'),
    std_z      = ('redshift', 'std'),
    mean_r_mag = ('r_mag', 'mean'),
)
pd.set_option('display.width', 120)
print("\n", summary.round(3))
print("\n'size' counts rows, 'count' counts non-missing values: "
      f"{summary['count'].sum()} rows, {summary['n_z'].sum()} redshifts")
