"""Ch.4  Build the data files used in this chapter (all SYNTHETIC, SEED = 2026).

  data/raw/co60_counts.csv            1,000 one-second counts, Poisson with mean 5
                                      (exactly the recipe of exercise E4.2)
  data/raw/halpha_spectrum.csv        an H-alpha line: Gaussian, sigma = 0.020 nm
                                      (Doppler width at T = 1e4 K) + baseline + noise
  data/processed/galaxy10_features.csv
                                      the Chapter 3 galaxy table (stars removed, g_mag
                                      numeric) plus a light 'concentration' feature C
                                      (tab:invariant), drawn per class: smooth galaxies
                                      C ~ 4, spirals C ~ 2.8
Chapter 3 must have been run once (its 01_make_raw_data.py); if not, it is run here.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.constants as const

SEED = 2026
HERE = Path(__file__).resolve().parent
os.chdir(HERE)
rng = np.random.default_rng(SEED)
Path('data/raw').mkdir(parents=True, exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)

# --- Co-60 counts: exactly the recipe of E4.2 ---------------------------------
counts = rng.poisson(lam=5, size=1000)
pd.DataFrame({'interval': np.arange(1000), 'counts': counts}).to_csv(
    'data/raw/co60_counts.csv', index=False)

# --- H-alpha line with Doppler width at 1e4 K ---------------------------------
lam0 = 656.28                                    # nm
m_H = const.m_p + const.m_e
sigma_true = lam0 * np.sqrt(const.k * 1e4 / (m_H * const.c**2))
wavelength = np.linspace(lam0 - 0.25, lam0 + 0.25, 201)
intensity = (100 * np.exp(-(wavelength - lam0)**2 / (2 * sigma_true**2)) + 10
             + rng.normal(0, 2.0, wavelength.size))
pd.DataFrame({'wavelength_nm': np.round(wavelength, 5),
              'intensity': np.round(intensity, 3)}).to_csv('data/raw/halpha_spectrum.csv',
                                                           index=False)

# --- Galaxy10-style features ----------------------------------------------------
ch3 = HERE.parent / 'ch03_cleaning'
meta = ch3 / 'data' / 'raw' / 'galaxy10_metadata.csv'
if not meta.exists():
    subprocess.run([sys.executable, str(ch3 / '01_make_raw_data.py')], check=True)
df = pd.read_csv(meta)
df['g_mag'] = pd.to_numeric(df['g_mag'], errors='coerce')
df = df[df['r_mag'] >= 8].copy()                 # the five foreground stars (Ch.3)
C_mean = {'Round Smooth': 4.1, 'In-between Round Smooth': 3.8, 'Cigar Shaped Smooth': 3.6,
          'Edge-on with Bulge': 3.3, 'Edge-on without Bulge': 2.8, 'Barred Spiral': 3.0,
          'Unbarred Tight Spiral': 2.9, 'Unbarred Loose Spiral': 2.7,
          'Disturbed': 2.8, 'Merging': 2.9}
df['concentration'] = np.round(df['galaxy_class'].map(C_mean) + rng.normal(0, 0.3, len(df)), 3)
df.to_csv('data/processed/galaxy10_features.csv', index=False)

print(f"co60_counts.csv        : {counts.size} counts, mean {counts.mean():.3f}")
print(f"halpha_spectrum.csv    : sigma used {sigma_true:.4f} nm (T = 1e4 K)")
print(f"galaxy10_features.csv  : {len(df)} galaxies, columns {list(df.columns)}")
