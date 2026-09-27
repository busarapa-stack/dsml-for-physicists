"""Ch.7  Toy AFM force map: K-Means, GMM and Ward (subsample + nearest centroid), k = 2..5,
winner-takes-all and soft rules, F1 / MCC against the reference map.

Book references: sec:afm-case, ssec:afm-PI, code:afm-3algo (line for line), tab:afm-results
The map is SYNTHETIC (seed 2026).  It imitates the design of the author's AFM study; no unpublished data are used.
The listing continues the notebook: KMeans / np (01), GaussianMixture (03), silhouette_score (04) are executed first.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
Path('outputs').mkdir(exist_ok=True)
for _f in ['01_kmeans_elbow.py', '03_gmm_polymer.py', '04_silhouette.py']:
    _src = (HERE / _f).read_text()
    exec(_src[_src.index('\n# --- code:'):_src.index('\n# --- end of listings')])   # earlier listings

# --- code:afm-3algo ----------------------------------------------------------
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import f1_score, matthews_corrcoef

def make_scaffold_map(n=128, seed=2026):
    """Toy AFM force map: scaffold (label 1) vs substrate (paraffin or glass).
    Channels: height H (nm), contact stiffness S (N/m), adhesion A (nN)."""
    rng = np.random.default_rng(seed)
    phi = gaussian_filter(rng.normal(size=(n, n)), sigma=6); phi /= phi.std()
    scaffold = phi > -0.55                                    # about 70 % scaffold
    g = gaussian_filter(rng.normal(size=(n, n)), sigma=4); g /= g.std()
    glass = (~scaffold) & (g > 0.8)                           # exposed glass
    het = gaussian_filter(rng.normal(size=(n, n)), sigma=3); het /= het.std()
    H = np.where(scaffold, 250 + 72 * het, np.where(glass, 20.0, 130.0))
    S = np.where(scaffold, 0.8 + 0.18 * het, np.where(glass, 1.5, 0.5))
    A = np.where(scaffold, 4.5 - 0.72 * het, np.where(glass, 2.5, 6.5))
    H, S, A = (gaussian_filter(m, sigma=1.0) for m in (H, S, A))   # tip blur
    heavy = lambda s: s * rng.standard_t(df=3, size=(n, n))        # heavy tails
    H += heavy(30.0); S += heavy(0.09); A += heavy(0.65)
    X = np.column_stack([H.ravel(), S.ravel(), A.ravel()])
    return X, scaffold.ravel().astype(int), glass.ravel()

X_afm, y_afm, glass_afm = make_scaffold_map()
Xa = RobustScaler().fit_transform(X_afm)

def ward_labels(k, n_sub=5000, seed=42):
    """Ward on a subsample, then assign every pixel to the nearest centroid."""
    idx = np.random.default_rng(seed).choice(len(Xa), n_sub, replace=False)
    lab = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Xa[idx])
    return NearestCentroid().fit(Xa[idx], lab).predict(Xa)

algorithms = {
    'K-Means': lambda k: KMeans(k, n_init=20, random_state=42).fit_predict(Xa),
    'GMM':     lambda k: GaussianMixture(k, covariance_type='full', n_init=10,
                                         random_state=42).fit(Xa).predict(Xa),
    'Ward':    ward_labels,
}
sub = np.random.default_rng(0).choice(len(Xa), 5000, replace=False)
for name, fit in algorithms.items():
    for k in range(2, 6):
        lab = fit(k)
        share = {c: y_afm[lab == c].mean() for c in np.unique(lab)}  # evaluation only
        wta  = (lab == max(share, key=share.get)).astype(int)
        soft = np.isin(lab, [c for c, v in share.items() if v >= 0.5]).astype(int)
        print(f"{name:8s} k={k} sil={silhouette_score(Xa[sub], lab[sub]):.3f} "
              f"WTA F1={f1_score(y_afm, wta):.3f} MCC={matthews_corrcoef(y_afm, wta):.3f} "
              f"soft F1={f1_score(y_afm, soft):.3f} MCC={matthews_corrcoef(y_afm, soft):.3f}")
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
    print(f"\nscaffold fraction {y_afm.mean():.3f}, exposed glass {glass_afm.mean():.3f}, "
          f"paraffin {1 - y_afm.mean() - glass_afm.mean():.3f}")
    for j, ch in enumerate('HSA'):
        dev = np.abs(X_afm[:, j] - np.median(X_afm[:, j]))
        print(f"{ch}: 99.9th percentile of |x - median| / median of |x - median| = "
              f"{np.percentile(dev, 99.9) / np.median(dev):.1f}  (Gaussian: 4.9)")

    print("\nDBSCAN on the full scaled map (min_samples=5):")
    for eps in (0.3, 0.5):
        lab = DBSCAN(eps=eps, min_samples=5).fit_predict(Xa)
        sizes = np.sort(np.bincount(lab[lab >= 0]))[::-1]
        print(f"  eps={eps}: largest cluster {sizes[0]} of {len(Xa)} pixels ({100 * sizes[0] / len(Xa):.1f} %), "
              f"other clusters have at most {sizes[1] if len(sizes) > 1 else 0} pixels, noise {(lab == -1).sum()}")

    print("\nK-Means clusters in physical units (median H nm, S N/m, A nN):")
    for k in (2, 3):
        lab = KMeans(k, n_init=20, random_state=42).fit_predict(Xa)
        for c in np.argsort([np.median(X_afm[lab == c, 0]) for c in range(k)]):
            m = lab == c
            print(f"  k={k} cluster: {100 * m.mean():4.1f} % of pixels, scaffold {100 * y_afm[m].mean():4.1f} %, "
                  f"glass {100 * glass_afm[m].mean():4.1f} %, median (H, S, A) = {np.median(X_afm[m], axis=0).round(2)}")

    print("\ninternal indices (K-Means; GMM for BIC):")
    for k in range(2, 7):
        lab = KMeans(k, n_init=20, random_state=42).fit_predict(Xa)
        bic = GaussianMixture(k, covariance_type='full', n_init=10, random_state=42).fit(Xa).bic(Xa)
        print(f"  k={k} silhouette {silhouette_score(Xa[sub], lab[sub]):.3f}  "
              f"Davies-Bouldin {davies_bouldin_score(Xa, lab):.3f}  "
              f"Calinski-Harabasz {calinski_harabasz_score(Xa, lab):.0f}  GMM BIC {bic:.0f}")

    lab3 = KMeans(3, n_init=20, random_state=42).fit_predict(Xa)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, img, title in zip(axes, [X_afm[:, 0], X_afm[:, 1], X_afm[:, 2], y_afm + glass_afm * 0.5, lab3],
                              ['H (nm)', 'S (N/m)', 'A (nN)', 'reference (1 scaffold, 0.5 glass)', 'K-Means k=3']):
        img = np.asarray(img, float)
        lo, hi = np.percentile(img, [1, 99])      # heavy tails would wash out the colour scale
        im = ax.imshow(img.reshape(128, 128), cmap='viridis', vmin=lo, vmax=hi); ax.set_title(title); ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig('outputs/ch07_afm_maps.png', dpi=90)
    print("saved outputs/ch07_afm_maps.png")
