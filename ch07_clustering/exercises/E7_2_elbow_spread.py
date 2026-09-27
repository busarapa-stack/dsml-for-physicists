"""E7.2  Elbow and silhouette for cluster_std = 1.0 (code:kmeans) and 2.5."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE); Path('outputs').mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for row, std in enumerate([1.0, 2.5]):
    X, _ = make_blobs(n_samples=600, centers=3, cluster_std=std, random_state=2026)
    X_scaled = StandardScaler().fit_transform(X)
    W = [KMeans(k, n_init=10, random_state=42).fit(X_scaled).inertia_ for k in range(1, 11)]
    sil = [silhouette_score(X_scaled, KMeans(k, n_init=10, random_state=42).fit_predict(X_scaled))
           for k in range(2, 8)]
    k_star = 2 + int(np.argmax(sil))
    print(f"cluster_std = {std}: W(K=1..6) = {[round(w) for w in W[:6]]}")
    print(f"   silhouette K=2..7 = {[round(s, 3) for s in sil]} -> K* = {k_star}")
    km = KMeans(k_star, n_init=10, random_state=42).fit(X_scaled)
    axes[row, 0].plot(range(1, 11), W, 'o-'); axes[row, 0].set_title(f'std {std}: W vs K')
    axes[row, 1].plot(range(2, 8), sil, 'o-'); axes[row, 1].set_title('silhouette vs K')
    axes[row, 2].scatter(*X_scaled.T, c=km.labels_, s=6); axes[row, 2].scatter(*km.cluster_centers_.T, c='red', marker='x', s=100)
    axes[row, 2].set_title(f'K* = {k_star}')
fig.tight_layout(); fig.savefig('outputs/E7_2_elbow_spread.png', dpi=90)
print("saved outputs/E7_2_elbow_spread.png")
