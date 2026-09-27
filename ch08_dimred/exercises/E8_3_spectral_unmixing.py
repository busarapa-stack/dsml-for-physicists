"""E8.3  Spectral unmixing of 40 mixtures of three pure spectra: FastICA vs PCA vs NMF.

Pure spectra (Gaussian peaks on 500 points, 400-800 nm):
  s1 = peak 480 nm (width 12) + 0.3 x peak 620 nm (15);  s2 = 530 nm (20) + 0.5 x 700 nm (12);  s3 = 590 nm (25).
Concentrations from rng.dirichlet (they sum to one), noise sd 0.01.  Ten data sets (seeds 0-9).
The run with concentrations multiplied by a random total amount (0.5-1.5) shows the effect of closure.
"""
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF, PCA, FastICA

warnings.filterwarnings('ignore')
wl = np.linspace(400, 800, 500)
G = lambda c, w: np.exp(-0.5 * ((wl - c) / w)**2)
S = np.array([G(480, 12) + 0.3 * G(620, 15), G(530, 20) + 0.5 * G(700, 12), G(590, 25)])
print("correlation between the pure spectra:", np.corrcoef(S)[np.triu_indices(3, 1)].round(2))


def matched(E):
    C = np.abs(np.corrcoef(S, E)[:3, 3:])
    r, c = linear_sum_assignment(-C)
    return np.sort(C[r, c])


for closure in (True, False):
    res = {'FastICA': [], 'PCA': [], 'NMF': []}; share3 = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        conc = rng.dirichlet(np.ones(3), size=40)
        if not closure:
            conc *= rng.uniform(0.5, 1.5, size=(40, 1))
        X = conc @ S + 0.01 * rng.normal(size=(40, 500))
        share3.append(PCA(3).fit(X).explained_variance_ratio_[2])
        res['FastICA'].append(matched(FastICA(3, whiten='unit-variance', random_state=42, max_iter=2000).fit_transform(X.T).T))
        res['PCA'].append(matched(PCA(3).fit(X).components_))
        res['NMF'].append(matched(NMF(3, init='nndsvda', max_iter=2000, random_state=42).fit(np.clip(X, 0, None)).components_))
    print(f"\nconcentrations {'sum to one (Dirichlet, as in the exercise)' if closure else 'x random total amount'}: "
          f"3rd PCA component carries {100 * np.mean(share3):.2f} % of the variance")
    for k, v in res.items():
        v = np.array(v)
        print(f"  {k:8s} |corr| range {v.min():.3f}-{v.max():.3f}; median per component (worst..best) {np.median(v, axis=0).round(3)}")

rng = np.random.default_rng(0)
X = rng.dirichlet(np.ones(3), size=40) @ S + 0.01 * rng.normal(size=(40, 500))
for name, E in [('FastICA', FastICA(3, whiten='unit-variance', random_state=42, max_iter=2000).fit_transform(X.T).T),
                ('PCA', PCA(3).fit(X).components_)]:
    E = E * np.sign(E[np.arange(3), np.abs(E).argmax(axis=1)])[:, None]     # make the largest excursion positive
    print(f"{name}: most negative value / largest value of each component (seed 0): {(E.min(axis=1) / E.max(axis=1)).round(2)}")
