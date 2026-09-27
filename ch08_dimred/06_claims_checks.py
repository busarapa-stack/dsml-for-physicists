"""Ch.8  Checks of statements in the text that have no listing of their own (second pass).

  1  all six listings run in book order as one notebook (X of code:pca = 21 AFM features)
  2  probabilistic PCA: sigma^2 = mean of the discarded eigenvalues                 (ssec:ppca-uq)
  3  standardising the polymer coordinates: does PCA still find the Rouse modes?   (ssec:pca-algo, review Q3)
  4  forgetting to remove centre-of-mass motion: PC1 becomes a rigid translation   (notebox after thm:pca-modes)
  5  independent samples of mode 1 ~ T / (2 tau_1); spread inside the first triplet (ssec:rouse-modes)
  6  shape of mode 1: the chain ends move most                                      (ssec:rouse-modes)
  7  run time of t-SNE and UMAP on 3,000 and 16,384 points                           (ssec:umap, instructor manual)
About 2-3 minutes.
"""
import os
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
warnings.filterwarnings('ignore')

# 1 ------------------------------------------------------------------------------
t0 = time.time()
nb = {}
for f in ('01_pca_bootstrap.py', '02_ica_cocktail.py', '03_tsne_umap_afm.py', '04_rouse_pca.py'):
    s = (HERE / f).read_text()
    nb['__file__'] = str(HERE / f)
    pre = s[s.index('\nimport'):s.index('\n# --- code:')]
    exec(pre.replace('os.chdir(', '(lambda *a: None)('), nb)          # preambles (imports, X for code:pca)
    exec(s[s.index('\n# --- code:'):s.index('\n# --- end of listings')], nb)
print(f"1  six listings ran in book order as one notebook ({time.time() - t0:.0f} s)")

import numpy as np
from sklearn.decomposition import PCA

# 2 ------------------------------------------------------------------------------
Xs = nb['X_scaled']
for d in (3, 5):
    p = PCA(n_components=d).fit(Xs)
    full = PCA().fit(Xs).explained_variance_
    print(f"2  d={d}: sklearn noise_variance_ {p.noise_variance_:.4f}, mean of discarded eigenvalues {full[d:].mean():.4f}")

# 3 ------------------------------------------------------------------------------
Xc, N = nb['X_centered'], 50
i = np.arange(N)
std = Xc / Xc.std(axis=0)
per_bead = Xc.reshape(-1, N, 3).var(axis=0).sum(axis=1)
print(f"3  per-bead variance: ends {per_bead[[0, -1]].round(1)}, middle {per_bead[N // 2]:.1f} (ratio {per_bead[0] / per_bead[N // 2]:.1f})")
for name, Z in (('centred only', Xc), ('standardised', std)):
    p = PCA(9).fit(Z)
    m = p.components_[0].reshape(N, 3); prof = m @ np.linalg.svd(m)[2][0]
    r = abs(np.corrcoef(prof, np.cos(np.pi * (i + .5) / N))[0, 1])
    g = p.explained_variance_
    print(f"   {name:13s}: |corr| PC1 with cos(p=1) {r:.4f}; triplet means {g[:3].mean():.2f} / {g[3:6].mean():.2f} / {g[6:9].mean():.2f}; "
          f"ratio p1/p2 {g[:3].mean() / g[3:6].mean():.2f} (theory {np.sin(np.pi / 50)**2 / np.sin(np.pi / 100)**2:.2f})")

# 4 ------------------------------------------------------------------------------
rng = np.random.default_rng(2026)
cm = np.cumsum(rng.normal(scale=np.sqrt(2 * 5.0 / N), size=(len(Xc), 3)), axis=0)   # free diffusion of the centre of mass
Xcm = nb['polymer_trajectory'] + np.tile(cm, N)
p = PCA(3).fit(Xcm - Xcm.mean(axis=0))
v = p.components_[0].reshape(N, 3)
print(f"4  with centre-of-mass diffusion kept: PC1 carries {p.explained_variance_ratio_[0]:.2f} of the variance; "
      f"spread of its 50 bead vectors relative to their mean {np.linalg.norm(v - v.mean(0)) / np.linalg.norm(v):.3f} (0 = rigid translation)")

# 5 ------------------------------------------------------------------------------
tau1 = nb['tau_theory'][0]
ev = nb['pca'].explained_variance_
print(f"5  T/(2 tau_1) = {10000 * 5.0 / (2 * tau1):.0f} independent samples of mode 1; "
      f"spread inside first triplet (max - min)/mean = {(ev[0] - ev[2]) / ev[:3].mean():.2f}")

# 6 ------------------------------------------------------------------------------
prof = nb['profile']
print(f"6  |PC1 displacement| at beads 1, 25, 50: {np.abs(prof[[0, 24, 49]]).round(3)} (ends largest: {np.argmax(np.abs(prof)) in (0, 49)})")

# 7 ------------------------------------------------------------------------------
import umap
from sklearn.manifold import TSNE
F = nb['F']
for n in (3000, len(F)):
    Z = F[np.random.default_rng(0).choice(len(F), n, replace=False)]
    t0 = time.time(); TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(Z); tt = time.time() - t0
    t0 = time.time(); umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(Z); tu = time.time() - t0
    print(f"7  {n:5d} points: t-SNE {tt:.0f} s, UMAP {tu:.0f} s (UMAP with random_state runs single-threaded; first call includes numba compilation)")
