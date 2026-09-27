"""E8.4  PCA / t-SNE / UMAP of the 21 AFM features: trustworthiness, 5-NN accuracy, perplexity and n_neighbors,
and K-Means (k = 3) MCC with 21 features vs the three raw channels.  About 3-4 minutes."""
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
warnings.filterwarnings('ignore')
__file__ = str(HERE / '03_tsne_umap_afm.py')
_s = (HERE / '03_tsne_umap_afm.py').read_text()
exec(_s[:_s.index('\n# --- end of listings')].split('"""', 2)[2])      # builds F, Z, labels, emb

from sklearn.cluster import KMeans
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
_, _, glass = make_scaffold_map(); gl = glass[idx]
runs = [('t-SNE', p, TSNE(n_components=2, perplexity=p, random_state=42)) for p in (5, 30, 100)] + \
       [('UMAP', k, umap.UMAP(n_components=2, n_neighbors=k, min_dist=0.1, random_state=42)) for k in (5, 15, 100)]
for ax, (name, par, model) in zip(axes.ravel(), runs):
    E = model.fit_transform(Z)
    acc = cross_val_score(KNeighborsClassifier(n_neighbors=5), E, labels, cv=5).mean()
    print(f"{name:5s} {'perplexity' if name == 't-SNE' else 'n_neighbors'}={par:3d}: trustworthiness "
          f"{trustworthiness(Z, E, n_neighbors=10):.3f}, 5-NN accuracy {acc:.3f}")
    ax.scatter(*E[labels == 1].T, s=2, c='tab:orange'); ax.scatter(*E[(labels == 0) & ~gl].T, s=2, c='tab:blue')
    ax.scatter(*E[gl].T, s=2, c='k'); ax.set_title(f"{name} {par}")
fig.tight_layout(); fig.savefig('outputs/E8_4_params.png', dpi=80)
print("saved outputs/E8_4_params.png")


def mcc_wta(lab):
    share = {c: y_afm[lab == c].mean() for c in np.unique(lab)}
    return matthews_corrcoef(y_afm, (lab == max(share, key=share.get)).astype(int))


print(f"K-Means k=3 MCC (winner-take-all): 21 features {mcc_wta(KMeans(3, n_init=20, random_state=42).fit_predict(F)):.3f}, "
      f"3 channels {mcc_wta(KMeans(3, n_init=20, random_state=42).fit_predict(RobustScaler().fit_transform(X_afm))):.3f}")
acc21 = cross_val_score(KNeighborsClassifier(n_neighbors=5), Z, labels, cv=5).mean()
acc3 = cross_val_score(KNeighborsClassifier(n_neighbors=5), RobustScaler().fit_transform(X_afm)[idx], labels, cv=5).mean()
print(f"5-NN accuracy on the same 3,000 pixels: 21 features {acc21:.3f}, 3 channels {acc3:.3f}")
