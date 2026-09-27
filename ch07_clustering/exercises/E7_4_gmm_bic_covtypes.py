"""E7.4  BIC for K = 1..6 and the four covariance types at K*, compared with K-Means (polymer data of code:gmm-polymer)."""
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
__file__ = str(HERE / '03_gmm_polymer.py')
src = (HERE / '03_gmm_polymer.py').read_text()
exec(src[:src.index('\n# --- end of listings')].split('"""', 2)[2])     # builds X_polymer, state_true

bic = [GaussianMixture(k, covariance_type='full', n_init=10, random_state=42).fit(X_polymer).bic(X_polymer)
       for k in range(1, 7)]
k_star = 1 + int(np.argmin(bic))
print("BIC K=1..6:", [round(b) for b in bic], "-> K* =", k_star)
print("true state fractions:", np.bincount(state_true).round() / len(state_true))
for ct in ['full', 'diag', 'tied', 'spherical']:
    m = GaussianMixture(k_star, covariance_type=ct, n_init=10, random_state=42).fit(X_polymer)
    print(f"{ct:9s}: BIC {m.bic(X_polymer):7.0f}  ARI {adjusted_rand_score(state_true, m.predict(X_polymer)):.3f}  "
          f"weights {np.sort(m.weights_).round(3)}")
lab = KMeans(3, n_init=10, random_state=42).fit_predict(StandardScaler().fit_transform(X_polymer))
print(f"K-Means  : ARI {adjusted_rand_score(state_true, lab):.3f}  cluster fractions {np.sort(np.bincount(lab) / len(lab)).round(3)}")
