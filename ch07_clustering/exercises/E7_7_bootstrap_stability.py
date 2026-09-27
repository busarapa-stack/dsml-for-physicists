"""E7.7  Bootstrap stability of K-Means (k = 2..5) and GMM (k = 3) on the SYNTHETIC AFM map of code:afm-3algo.

Data resampling and algorithm initialisation use different seeds (DATA_SEED vs 1000 + b).
    python E7_7_bootstrap_stability.py            # DATA_SEED = 2026, about 2 minutes
    python E7_7_bootstrap_stability.py --seeds    # repeat for data seeds 0, 1, 2026 (about 6 minutes)
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE); Path('outputs').mkdir(exist_ok=True)
for _f in ['01_kmeans_elbow.py', '03_gmm_polymer.py', '04_silhouette.py']:
    _s = (HERE / _f).read_text(); exec(_s[_s.index('\n# --- code:'):_s.index('\n# --- end of listings')])
_s = (HERE / '05_afm_three_algorithms.py').read_text()
_s = _s[_s.index('\n# --- code:afm'):_s.index('\n# --- end of listings')]
exec(_s[:_s.index('sub = np.random')])

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix

N_BOOT = 20


def align(ref, lab, k):
    """Relabel lab so that its clusters match ref as well as possible (Hungarian matching)."""
    C = confusion_matrix(ref, lab, labels=range(k))
    r, c = linear_sum_assignment(-C)
    return np.array([dict(zip(c, r))[v] for v in range(k)])[lab]


def model(name, k, seed):
    if name == 'K-Means':
        return KMeans(k, n_init=20, random_state=seed)
    return GaussianMixture(k, covariance_type='full', n_init=10, random_state=seed)


def stability(data_seed, show=False):
    rng = np.random.default_rng(data_seed)
    boots = [rng.choice(len(Xa), len(Xa), replace=True) for _ in range(N_BOOT)]
    unstable_maps = {}
    for name, k in [('K-Means', 2), ('K-Means', 3), ('K-Means', 4), ('K-Means', 5), ('GMM', 3)]:
        ref = model(name, k, 42).fit(Xa).predict(Xa)
        ari, same = [], np.zeros(len(Xa))
        for b, idx in enumerate(boots):
            lab = model(name, k, 1000 + b).fit(Xa[idx]).predict(Xa)
            ari.append(adjusted_rand_score(ref, lab))
            same += align(ref, lab, k) == ref
        unstable = same < 18
        unstable_maps[(name, k)] = (unstable, ref)
        print(f"  data seed {data_seed:4d}  {name:7s} k={k}: ARI mean {np.mean(ari):.3f}  min {np.min(ari):.3f}  "
              f"unstable pixels {100 * unstable.mean():.1f} %", flush=True)
    return unstable_maps


maps = stability(2026)
un, ref = maps[('K-Means', 3)]
yy = y_afm.reshape(128, 128)
edge = np.zeros_like(yy, bool)
for sh in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
    edge |= np.roll(yy, sh, (0, 1)) != yy
edge = edge.ravel()
order = np.argsort([np.median(X_afm[ref == c, 0]) for c in range(3)])
names = dict(zip(order, ['glass-like (low)', 'low mixed', 'high scaffold']))
print(f"\nK-Means k=3 unstable pixels: {un.sum()}; on a scaffold/substrate edge in the image: {100 * edge[un].mean():.0f} % "
      f"(all pixels: {100 * edge.mean():.0f} %)")
print("  reference cluster of the unstable pixels:", {names[c]: int((ref[un] == c).sum()) for c in range(3)})
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].imshow(y_afm.reshape(128, 128)); ax[0].set_title('reference')
ax[1].imshow(un.reshape(128, 128), cmap='Reds'); ax[1].set_title('K-Means k=3: same cluster < 18/20')
for a in ax: a.axis('off')
fig.tight_layout(); fig.savefig('outputs/E7_7_unstable_pixels.png', dpi=90)
print("saved outputs/E7_7_unstable_pixels.png")

if '--seeds' in sys.argv:
    for s in (0, 1):
        stability(s)
