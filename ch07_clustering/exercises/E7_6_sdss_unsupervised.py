"""E7.6  Cluster SDSS objects on four colour indices only (no redshift, no labels), then compare with the labels.

Uses the SDSS table of Chapter 6: ../ch06_classification/data/raw/star_classification.csv
(the real Kaggle file if the reader put it there, otherwise ch06's SYNTHETIC stand-in is created).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
CH6 = HERE.parent / 'ch06_classification'
path = CH6 / 'data/raw/star_classification.csv'
if not path.exists():
    subprocess.run([sys.executable, str(CH6 / '05_sdss_data.py')], check=True)
print((CH6 / 'data/raw/SOURCE.txt').read_text().strip())

df = pd.read_csv(path)
bands = ['u', 'g', 'r', 'i', 'z']
df = df[(df[bands] > -100).all(axis=1)]
C = np.column_stack([df.u - df.g, df.g - df.r, df.r - df.i, df.i - df.z])   # colour indices as in code:sdss-cc
y = df['class'].to_numpy()
Z = StandardScaler().fit_transform(C)

km = KMeans(3, n_init=10, random_state=42).fit_predict(Z)
print(f"\nK-Means K=3: ARI {adjusted_rand_score(y, km):.3f}")
print(pd.crosstab(km, y))

sub = np.random.default_rng(0).choice(len(Z), 20000, replace=False)     # BIC on a 20,000-object subsample
bic = [GaussianMixture(k, covariance_type='full', n_init=3, random_state=42).fit(Z[sub]).bic(Z[sub]) for k in range(1, 11)]
k_star = 1 + int(np.argmin(bic))
print("\nGMM BIC K=1..10:", [round(b) for b in bic], "-> K* =", k_star)
gm = GaussianMixture(k_star, covariance_type='full', n_init=3, random_state=42).fit(Z[sub]).predict(Z)
print(f"GMM K={k_star}: ARI {adjusted_rand_score(y, gm):.3f}")
tab = pd.crosstab(gm, y)
tab['u-g'] = pd.Series(C[:, 0]).groupby(gm).median().round(2)
tab['g-r'] = pd.Series(C[:, 1]).groupby(gm).median().round(2)
print(tab)
q = tab['QSO'] / tab[['GALAXY', 'QSO', 'STAR']].sum(axis=1)
best = q.idxmax()
print(f"most quasar-rich GMM cluster: {100 * q[best]:.0f} % QSO, holds {100 * tab.loc[best, 'QSO'] / (y == 'QSO').sum():.0f} % of all QSOs, "
      f"median u-g {tab.loc[best, 'u-g']}")
