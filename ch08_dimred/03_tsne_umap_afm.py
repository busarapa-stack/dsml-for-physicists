"""Ch.8  21 local features of the AFM map, and 2-D views with PCA, t-SNE and UMAP.

Book references: ssec:umap, code:tsne-umap (line for line), ssec:dr-quality
Needs umap-learn.  make_scaffold_map() is the Chapter 7 function (SYNTHETIC map, seed 2026).
t-SNE and UMAP take about 30-60 s.  UMAP results can differ slightly between machines (numba).
"""
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
Path('outputs').mkdir(exist_ok=True)
warnings.filterwarnings('ignore')
_s = (HERE.parent / 'ch07_clustering' / '05_afm_three_algorithms.py').read_text()
_s = _s[_s.index('\n# --- code:afm'):]
exec(_s[:_s.index('X_afm, y_afm, glass_afm = make_scaffold_map()')])     # make_scaffold_map() of Chapter 7

# --- code:tsne-umap ----------------------------------------------------------
from scipy.ndimage import uniform_filter
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE, trustworthiness
import umap                                   # pip install umap-learn

X_afm, y_afm, _ = make_scaffold_map()         # function from the Chapter 7 listing
n = 128
features = []
for ch in range(3):                           # H, S, A
    m = X_afm[:, ch].reshape(n, n)
    features.append(m)
    for w in (3, 7, 15):                      # local mean and local std
        mu = uniform_filter(m, w)
        sd = np.sqrt(np.maximum(uniform_filter(m * m, w) - mu * mu, 0))
        features += [mu, sd]
F = RobustScaler().fit_transform(np.column_stack([f.ravel() for f in features]))

idx = np.random.default_rng(0).choice(len(F), 3000, replace=False)
Z, labels = F[idx], y_afm[idx]
emb = {'PCA':  PCA(n_components=2).fit_transform(Z),
       't-SNE': TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(Z),
       'UMAP': umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                         random_state=42).fit_transform(Z)}
for name, E in emb.items():
    print(name, "trustworthiness =", round(trustworthiness(Z, E, n_neighbors=10), 3))
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    cum = np.cumsum(PCA().fit(Z).explained_variance_ratio_)
    print(f"PCA of the 21 features: 2 components {cum[1]:.3f}, 5 components {cum[4]:.3f} of the variance")
    print("5-NN accuracy (5-fold CV) predicting scaffold / substrate:")
    for name, E in list(emb.items()) + [('21 features', Z)]:
        acc = cross_val_score(KNeighborsClassifier(n_neighbors=5), E, labels, cv=5).mean()
        print(f"  {name:12s} {acc:.3f}")
    print(f"  all-scaffold guess {labels.mean():.3f} (this subsample; whole map {y_afm.mean():.3f})")
    _, _, glass = make_scaffold_map(); g = glass[idx]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, E) in zip(axes, emb.items()):
        ax.scatter(*E[labels == 1].T, s=3, c='tab:orange', label='scaffold')
        ax.scatter(*E[(labels == 0) & ~g].T, s=3, c='tab:blue', label='paraffin')
        ax.scatter(*E[g].T, s=3, c='k', label='glass'); ax.set_title(name)
    axes[0].legend(markerscale=4); fig.tight_layout(); fig.savefig('outputs/ch08_afm_embeddings.png', dpi=90)
    print("saved outputs/ch08_afm_embeddings.png")
