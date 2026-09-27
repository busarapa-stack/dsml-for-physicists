"""Ch.8  PCA with scikit-learn and bootstrap confidence intervals of the eigenvalues.

Book references: ssec:pca-algo, ssec:ppca-uq, code:pca, code:bootstrap-pca (line for line)
The book leaves the data matrix X of code:pca generic.  Here X is the 21 local AFM features of
code:tsne-umap (3 channels x [value + local mean/std in 3x3, 7x7, 15x15 windows], 16,384 pixels),
built from the SYNTHETIC Chapter 7 map.  The features have different units, so standardising is correct.
"""
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
_s = (HERE.parent / 'ch07_clustering' / '05_afm_three_algorithms.py').read_text()
_s = _s[_s.index('\n# --- code:afm'):]
exec(_s[:_s.index('X_afm, y_afm, glass_afm = make_scaffold_map()')])     # make_scaffold_map() of Chapter 7

_X_afm, _, _ = make_scaffold_map()
_maps = []
for _ch in range(3):
    _m = _X_afm[:, _ch].reshape(128, 128); _maps.append(_m)
    for _w in (3, 7, 15):
        _mu = uniform_filter(_m, _w)
        _maps += [_mu, np.sqrt(np.maximum(uniform_filter(_m * _m, _w) - _mu * _mu, 0))]
X = np.column_stack([_f.ravel() for _f in _maps])        # (16384, 21) in physical units

# --- code:pca ----------------------------------------------------------------
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)     # X: (n_samples, n_features)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)              # scores on the new axes
print("explained variance ratio:", pca.explained_variance_ratio_)
print("cumulative:", np.cumsum(pca.explained_variance_ratio_))
print("first component:", pca.components_[0])    # loadings on the original features
# --- code:bootstrap-pca ------------------------------------------------------
rng = np.random.default_rng(seed=2026)
n_bootstrap = 200
ev_boot = []
for _ in range(n_bootstrap):
    idx = rng.integers(0, len(X_scaled), size=len(X_scaled))   # resample rows
    ev_boot.append(PCA(n_components=5).fit(X_scaled[idx]).explained_variance_)
ev_boot = np.array(ev_boot)
ci_low, ci_high = np.percentile(ev_boot, [2.5, 97.5], axis=0)
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    print("\nbootstrap 95 % intervals of the first five eigenvalues (rows are independent pixels here,")
    print("but neighbouring pixels are correlated through the smoothing windows, so these are optimistic):")
    for k, (lo, hi, ev) in enumerate(zip(ci_low, ci_high, pca.explained_variance_[:5]), 1):
        print(f"  PC{k}: {ev:.2f}  [{lo:.2f}, {hi:.2f}]")
    print("Kaiser criterion (lambda > 1 on standardised data) keeps", int((pca.explained_variance_ > 1).sum()), "components")
