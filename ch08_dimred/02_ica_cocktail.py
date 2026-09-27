"""Ch.8  Blind source separation: FastICA vs PCA on three mixed signals.

Book references: ssec:ica-algo, code:ica (line for line)
numpy and PCA are imported in code:pca (01_pca_bootstrap.py); they are imported here directly.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

os.chdir(Path(__file__).resolve().parent)
Path('outputs').mkdir(exist_ok=True)

# --- code:ica ----------------------------------------------------------------
from scipy import signal
from sklearn.decomposition import FastICA

rng = np.random.default_rng(seed=2026)
t = np.linspace(0, 8, 2000)
S_true = np.c_[np.sin(2 * np.pi * 1.3 * t),              # three independent sources
               signal.square(2 * np.pi * 0.7 * t),
               signal.sawtooth(2 * np.pi * 0.45 * t)]
S_true += 0.05 * rng.normal(size=S_true.shape)
A = np.array([[1.0, 0.6, 0.4], [0.5, 1.0, 0.7], [0.3, 0.8, 1.0]])
X_mix = S_true @ A.T                                       # three observed mixtures

ica = FastICA(n_components=3, whiten='unit-variance', random_state=42)
S_ica = ica.fit_transform(X_mix)                           # recovered sources
S_pca = PCA(n_components=3).fit_transform(X_mix)           # for comparison
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    from scipy.optimize import linear_sum_assignment

    def matched_corr(S, E):
        C = np.abs(np.corrcoef(S.T, E.T)[:3, 3:])
        r, c = linear_sum_assignment(-C)
        return C[r, c]

    print("|correlation| with the true sources (matched):")
    print("  ICA:", matched_corr(S_true, S_ica).round(3))
    print("  PCA:", matched_corr(S_true, S_pca).round(3))
    fig, axes = plt.subplots(3, 3, figsize=(12, 6), sharex=True)
    for j in range(3):
        axes[j, 0].plot(t, S_true[:, j]); axes[j, 1].plot(t, S_ica[:, j]); axes[j, 2].plot(t, S_pca[:, j])
    for ax, title in zip(axes[0], ['true sources', 'FastICA', 'PCA']):
        ax.set_title(title)
    fig.tight_layout(); fig.savefig('outputs/ch08_ica_cocktail.png', dpi=90)
    print("saved outputs/ch08_ica_cocktail.png")
