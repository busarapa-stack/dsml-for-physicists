"""E8.6  Quality of 2-D embeddings of the Rouse trajectory (2,000 random frames): trustworthiness and continuity (k = 10),
PCA reconstruction error for d = 3, 9, 30, and linear regression of R_ee on the 2-D coordinates.  About 1-2 minutes."""
import os
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
warnings.filterwarnings('ignore')
_s = (HERE / '04_rouse_pca.py').read_text()
exec(_s[_s.index('import numpy'):_s.index('\n# --- end of listings')].replace("os.chdir(Path(__file__).resolve().parent)", ""))

import umap
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(2026)
sel = rng.choice(len(X_centered), 2000, replace=False)
Xs = X_centered[sel]
r = polymer_trajectory[sel].reshape(-1, N, 3)
ree = np.linalg.norm(r[:, -1] - r[:, 0], axis=1)

tot = ((Xs - Xs.mean(0))**2).sum()
for d in (2, 3, 9, 30):
    p = PCA(d).fit(Xs)
    rec = p.inverse_transform(p.transform(Xs))
    print(f"PCA d={d:2d}: relative reconstruction error {((Xs - rec)**2).sum() / tot:.3f}")

for seed in (42, 0):
    emb = {'PCA': PCA(2).fit_transform(Xs),
           't-SNE': TSNE(2, perplexity=30, random_state=seed).fit_transform(Xs),
           'UMAP': umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(Xs)}
    for name, E in emb.items():
        tw = trustworthiness(Xs, E, n_neighbors=10)
        co = trustworthiness(E, Xs, n_neighbors=10)            # continuity: roles swapped
        r2 = cross_val_score(LinearRegression(), E, ree, cv=5, scoring='r2').mean()
        print(f"seed {seed:2d} {name:5s}: trustworthiness {tw:.3f}  continuity {co:.3f}  CV R^2 of R_ee from 2-D coords {r2:.3f}")
from sklearn.neighbors import KNeighborsRegressor
cv = lambda model, feats, target: cross_val_score(model, feats, target, cv=5, scoring='r2').mean()
print("R_ee is a length: flipping the sign of a mode amplitude leaves it unchanged, so it is an even function of the scores.")
for name, E in emb.items():                                   # seed 0 embeddings from the loop above
    print(f"{name:5s} 2-D: CV R^2 of R_ee  linear {cv(LinearRegression(), E, ree):.3f}, "
          f"linear on (coords, coords^2) {cv(LinearRegression(), np.c_[E, E**2], ree):.3f}, 15-NN {cv(KNeighborsRegressor(15), E, ree):.3f}")
Rvec = r[:, -1] - r[:, 0]
for d in (3, 9, 30):
    Zd = PCA(d).fit_transform(Xs)
    print(f"PCA {d:2d} components: R_ee from squared scores {cv(LinearRegression(), Zd**2, ree):.3f}; "
          f"x-component of the R_ee vector, linear in the scores {cv(LinearRegression(), Zd, Rvec[:, 0]):.3f}")
# R_ee in terms of modes: R_ee vector = sum_p X_p (psi_p(N) - psi_p(1)), non-zero only for odd p
pp = np.arange(1, N)
w = np.sqrt(2 / N) * (np.cos(pp * np.pi * (N - 0.5) / N) - np.cos(pp * np.pi * 0.5 / N))
contrib = 3 * w**2 / (3 * 4 * np.sin(pp * np.pi / (2 * N))**2)      # 3 axes x w_p^2 <X_p^2>
print("share of <R_ee^2> from modes p = 1..6:", (contrib[:6] / contrib.sum()).round(3), "(even modes contribute zero);",
      f"theory <R_ee^2> = {contrib.sum():.1f} = (N-1) b^2 = {N - 1}; sample {np.mean(ree**2):.1f}")
