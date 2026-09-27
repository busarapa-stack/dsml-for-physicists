"""Ch.7  K-Means on three blobs and the elbow method.

Book references: sec:kmeans, ssec:choose-k, code:kmeans, code:elbow (line for line)
Data: make_blobs (SYNTHETIC), random_state 2026.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(Path(__file__).resolve().parent)
Path('outputs').mkdir(exist_ok=True)

# --- code:kmeans -------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=600, centers=3, cluster_std=1.0, random_state=2026)
X_scaled = StandardScaler().fit_transform(X)

pipe = Pipeline([
    ('scaler',  StandardScaler()),
    ('cluster', KMeans(n_clusters=3, init='k-means++', n_init=10,
                       random_state=42)),
])
labels = pipe.fit_predict(X)
centroids = pipe.named_steps['cluster'].cluster_centers_   # in scaled units
# --- code:elbow --------------------------------------------------------------
import numpy as np

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)          # inertia_ is W for this K
# plot range(1, 11) against wcss and look for the elbow
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    print("cluster sizes:", np.bincount(labels).tolist())
    print("centroids (scaled units):\n", centroids.round(2))
    print("W(K) for K = 1..10:", [round(w, 1) for w in wcss])
    print(f"W drops {wcss[0]:.0f} -> {wcss[1]:.0f} -> {wcss[2]:.1f}; "
          f"after K = 3 each extra cluster removes only {wcss[2] - wcss[3]:.1f} to {wcss[8] - wcss[9]:.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, s=8, cmap='viridis')
    axes[0].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=120)
    axes[0].set_title('K-Means, K = 3 (scaled units)')
    axes[1].plot(range(1, 11), wcss, 'o-'); axes[1].set_xlabel('K'); axes[1].set_ylabel('W (inertia)')
    axes[1].set_title('elbow at K = 3')
    fig.tight_layout(); fig.savefig('outputs/ch07_kmeans_elbow.png', dpi=100)
    print("saved outputs/ch07_kmeans_elbow.png")
