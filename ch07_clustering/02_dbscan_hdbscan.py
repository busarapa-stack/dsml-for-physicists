"""Ch.7  DBSCAN and HDBSCAN on two moons; chaining at a larger epsilon.

Book references: sec:dbscan, ssec:dbscan-algo, code:dbscan, code:hdbscan (line for line)
The listings continue the notebook of 01_kmeans_elbow.py (StandardScaler, KMeans), which is executed first.
Data: make_moons (SYNTHETIC), random_state 2026.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
Path('outputs').mkdir(exist_ok=True)
_src = (HERE / '01_kmeans_elbow.py').read_text()
exec(_src[_src.index('\n# --- code:'):_src.index('\n# --- end of listings')])   # earlier listings

# --- code:dbscan -------------------------------------------------------------
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X_moon, y_moon = make_moons(n_samples=500, noise=0.1, random_state=2026)
X_moon = StandardScaler().fit_transform(X_moon)

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_moon)          # label -1 means noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
# --- code:hdbscan ------------------------------------------------------------
from sklearn.cluster import HDBSCAN

clusterer = HDBSCAN(min_cluster_size=15, min_samples=5, copy=True)
labels_h = clusterer.fit_predict(X_moon)
prob_h = clusterer.probabilities_       # strength of membership, 0 for noise
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    from sklearn.metrics import adjusted_rand_score
    print(f"DBSCAN eps=0.3: {n_clusters} clusters, {n_noise} noise point(s), "
          f"ARI vs y_moon = {adjusted_rand_score(y_moon, labels):.3f}")
    lab35 = DBSCAN(eps=0.35, min_samples=5).fit_predict(X_moon)
    print(f"DBSCAN eps=0.35: {len(set(lab35)) - (-1 in lab35)} cluster(s) -> the two moons are chained together")
    km2 = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X_moon)
    print(f"K-Means K=2: ARI = {adjusted_rand_score(y_moon, km2):.3f} (straight cut through both moons)")
    nh = len(set(labels_h)) - (-1 in labels_h)
    print(f"HDBSCAN: {nh} clusters, {(labels_h == -1).sum()} noise points, "
          f"ARI = {adjusted_rand_score(y_moon, labels_h):.3f}, median membership strength {np.median(prob_h[labels_h >= 0]):.2f}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, lab, title in zip(axes, [km2, labels, lab35, labels_h],
                              ['K-Means K=2', 'DBSCAN eps=0.3', 'DBSCAN eps=0.35', 'HDBSCAN']):
        ax.scatter(X_moon[:, 0], X_moon[:, 1], c=lab, s=6, cmap='viridis')
        ax.scatter(*X_moon[lab == -1].T, c='red', marker='x', s=30)
        ax.set_title(title)
    fig.tight_layout(); fig.savefig('outputs/ch07_moons.png', dpi=100)
    print("saved outputs/ch07_moons.png")
