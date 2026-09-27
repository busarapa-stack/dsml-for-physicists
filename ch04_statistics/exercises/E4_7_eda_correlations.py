"""E4.7  The two most correlated pairs of features, with 95 % CIs by Fisher's z and by
bootstrap, and a check for a third variable behind the correlation.
"""
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)
df = pd.read_csv('data/processed/galaxy10_features.csv')
num = df[['redshift', 'r_mag', 'g_mag', 'concentration']]
pairs = sorted(combinations(num.columns, 2), key=lambda p: -abs(num[list(p)].dropna().corr().iloc[0, 1]))
rng = np.random.default_rng(47)
for a, b in pairs[:2]:
    d = num[[a, b]].dropna()
    r, n = d.corr().iloc[0, 1], len(d)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    boot = [d.iloc[rng.integers(0, n, n)].corr().iloc[0, 1] for _ in range(1000)]
    blo, bhi = np.percentile(boot, [2.5, 97.5])
    print(f"{a} vs {b}: r = {r:.3f} (n = {n}); Fisher 95 % CI [{lo:.3f}, {hi:.3f}]; "
          f"bootstrap [{blo:.3f}, {bhi:.3f}]")
d = df.dropna(subset=['redshift', 'r_mag', 'g_mag'])
res_r = d['r_mag'] - np.polyval(np.polyfit(d['redshift'], d['r_mag'], 1), d['redshift'])
res_g = d['g_mag'] - np.polyval(np.polyfit(d['redshift'], d['g_mag'], 1), d['redshift'])
print(f"\nr_mag vs g_mag after removing the common dependence on redshift: "
      f"r = {np.corrcoef(res_r, res_g)[0, 1]:.3f}")
print("-> distance (redshift) drives both magnitudes; the remaining correlation comes from"
      " each galaxy's own luminosity and colour. Missing redshifts are not random (Ch.3), so"
      " the redshift correlation describes only the brighter part of the sample.")
