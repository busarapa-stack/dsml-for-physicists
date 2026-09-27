"""E7.3  K-Means vs DBSCAN (eps 0.2-0.35) on the moons of code:dbscan: ARI, silhouette, k-distance plot."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE); Path('outputs').mkdir(exist_ok=True)

X_moon, y_moon = make_moons(n_samples=500, noise=0.1, random_state=2026)   # as in code:dbscan
X_moon = StandardScaler().fit_transform(X_moon)

km = KMeans(2, n_init=10, random_state=42).fit_predict(X_moon)
print(f"K-Means K=2    : ARI {adjusted_rand_score(y_moon, km):.3f}  silhouette {silhouette_score(X_moon, km):.3f}")
fig, axes = plt.subplots(1, 6, figsize=(22, 3.6))
axes[0].scatter(*X_moon.T, c=km, s=5); axes[0].set_title('K-Means')
for ax, eps in zip(axes[1:5], [0.2, 0.25, 0.3, 0.35]):
    lab = DBSCAN(eps=eps, min_samples=5).fit_predict(X_moon)
    nc = len(set(lab)) - (-1 in lab)
    ok = lab >= 0
    sil = f"{silhouette_score(X_moon[ok], lab[ok]):.3f}" if nc > 1 else "  -  "
    print(f"DBSCAN eps={eps:<4}: {nc} cluster(s), noise {100 * (~ok).mean():.1f} %, "
          f"ARI {adjusted_rand_score(y_moon, lab):.3f}  silhouette (noise excluded) {sil}")
    ax.scatter(*X_moon.T, c=lab, s=5); ax.set_title(f'DBSCAN eps={eps}')
kd = np.sort(NearestNeighbors(n_neighbors=6).fit(X_moon).kneighbors(X_moon)[0][:, 5])  # 5th neighbour (self excluded)
print(f"5th-neighbour distance: median {np.median(kd):.2f}, 95th pct {np.percentile(kd, 95):.2f}, "
      f"99th pct {np.percentile(kd, 99):.2f}, max {kd.max():.2f}")
axes[5].plot(kd); axes[5].set_title('k-distance (k = 5)'); axes[5].axhline(0.3, ls='--', c='gray')
fig.tight_layout(); fig.savefig('outputs/E7_3_moons.png', dpi=90)
print("saved outputs/E7_3_moons.png")
