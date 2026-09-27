"""E8.2  PCA of thermal fluctuations of a fixed-end harmonic chain (N = 20, m = k = k_B T = 1) vs analytic normal modes."""
import numpy as np
from sklearn.decomposition import PCA

N = 20
K = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
n = np.arange(1, N + 1)
j = np.arange(1, 6)
u = np.array([np.sqrt(2 / (N + 1)) * np.sin(jj * np.pi * n / (N + 1)) for jj in j])
theory = 1 / (4 * np.sin(j * np.pi / (2 * (N + 1)))**2)
print("theory 1/omega^2:", theory.round(1))
for seed in (2026, 0, 1):
    rng = np.random.default_rng(seed)
    Q = rng.multivariate_normal(np.zeros(N), np.linalg.inv(K), size=5000)
    pca = PCA().fit(Q)
    print(f"seed {seed:4d}: |<PC_j, u_j>| = {np.abs((pca.components_[:5] * u).sum(axis=1)).round(3)}  "
          f"eigenvalues {pca.explained_variance_[:5].round(1)}  PC1 share {pca.explained_variance_ratio_[0]:.3f}")
print(f"relative sampling error of an eigenvalue ~ sqrt(2/5000) = {np.sqrt(2 / 5000):.3f}")
