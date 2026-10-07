"""Ch.7  Second-pass checks of statements in the text that have no listing of their own.

  1  scikit-learn KMeans default n_init='auto' runs k-means++ only once          (sec:kmeans)
  2  chaining at eps = 0.35: core points whose neighbourhood touches both moons  (ssec:dbscan-algo)
  3  5th-neighbour distance plot: knee and percentiles                           (E7.3 answer)
  4  silhouette picks K = 2 when cluster_std = 4.0                               (E7.2 answer)
  5  Calinski-Harabasz on data with no clusters, D = 1..10                       (tab:4-metrics)
  6  gap statistic: three blobs and 20 uniform data sets                         (review question 6)
  7  per-pixel silhouette on the AFM map lies along cluster boundaries           (ssec:silhouette)
  8  sigmoid squashing before RobustScaler changes K-Means k = 3                 (ssec:afm-features)
  9  HDBSCAN on the AFM map                                                      (ssec:afm-compare)
 10  physics rule "highest mean H = scaffold" vs winner-take-all                 (ssec:afm-compare)
 11  Ward subsample (size and seed) sensitivity                                  (ssec:afm-compare, tab:afm-results)
 12  what GMM k = 2 separates                                                    (ssec:afm-PI, lesson 2)
Runs in about 3 minutes (item 6 is the slowest).
"""
import os
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
warnings.filterwarnings('ignore')
for _f in ['01_kmeans_elbow.py', '02_dbscan_hdbscan.py', '03_gmm_polymer.py', '04_silhouette.py']:
    _s = (HERE / _f).read_text(); exec(_s[_s.index('\n# --- code:'):_s.index('\n# --- end of listings')])
_s = (HERE / '05_afm_three_algorithms.py').read_text()
_s = _s[_s.index('\n# --- code:afm'):_s.index('\n# --- end of listings')]
exec(_s[:_s.index('sub = np.random')])             # AFM map, scaler and the three algorithms

import inspect
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors

print("1  KMeans n_init default:", inspect.signature(KMeans).parameters['n_init'].default,
      "-> runs actually used:", KMeans(3, random_state=0).fit(X_scaled)._n_init)

core = np.zeros(len(X_moon), bool)
for eps in (0.30, 0.35):
    db = DBSCAN(eps=eps, min_samples=5).fit(X_moon)
    core[:] = False; core[db.core_sample_indices_] = True
    nbrs = NearestNeighbors(radius=eps).fit(X_moon).radius_neighbors(X_moon[core], return_distance=False)
    print(f"2  eps={eps}: {sum(len(set(y_moon[n])) > 1 for n in nbrs)} of {core.sum()} core points have neighbours in both moons")

d = np.sort(NearestNeighbors(n_neighbors=6).fit(X_moon).kneighbors(X_moon)[0][:, 5])
x = np.linspace(0, 1, len(d)); yk = (d - d[0]) / (d[-1] - d[0])
print(f"3  5th-neighbour distance: knee (max distance below the chord) {d[np.argmax(x - yk)]:.2f}; "
      f"90/95/99th percentiles {np.percentile(d, [90, 95, 99]).round(2)}")

for std in (2.5, 3.0, 3.5, 4.0):
    Xb, _ = make_blobs(n_samples=600, centers=3, cluster_std=std, random_state=2026)
    Xb = StandardScaler().fit_transform(Xb)
    s = [silhouette_score(Xb, KMeans(k, n_init=10, random_state=42).fit_predict(Xb)) for k in (2, 3, 4)]
    print(f"4  cluster_std={std}: silhouette K=2,3,4 = {np.round(s, 3)}")

rng = np.random.default_rng(0)
for D in (1, 2, 3, 5, 10):
    U = rng.random((2000, D))
    print(f"5  uniform D={D:2d}: CH for K=2..7 =",
          [round(calinski_harabasz_score(U, KMeans(k, n_init=10, random_state=42).fit_predict(U))) for k in range(2, 8)])


def gap_k(Xd, kmax=6, B=20, seed=0):
    """Tibshirani et al. (2001): smallest K with Gap(K) >= Gap(K+1) - s(K+1); uniform box reference."""
    r = np.random.default_rng(seed); lo, hi = Xd.min(0), Xd.max(0); G, S = [], []
    for k in range(1, kmax + 1):
        w = np.log(KMeans(k, n_init=3, random_state=42).fit(Xd).inertia_)      # same n_init as the reference sets
        ref = [np.log(KMeans(k, n_init=3, random_state=42).fit(r.uniform(lo, hi, Xd.shape)).inertia_) for _ in range(B)]
        G.append(np.mean(ref) - w); S.append(np.std(ref) * np.sqrt(1 + 1 / B))
    return next((k for k in range(1, kmax) if G[k - 1] >= G[k] - S[k]), kmax)


print("6  gap statistic, blobs of code:kmeans -> K =", gap_k(X_scaled))
ks = [gap_k(np.random.default_rng(s).random((600, 2)), seed=100 + s) for s in range(20)]
print("   20 uniform data sets (600 points in a square) -> K chosen:", dict(zip(*np.unique(ks, return_counts=True))))

lab3 = KMeans(3, n_init=20, random_state=42).fit_predict(Xa)
L = lab3.reshape(128, 128); edge_c = np.zeros_like(L, bool)
for sh in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
    edge_c |= np.roll(L, sh, (0, 1)) != L
edge_c = edge_c.ravel()
neg = silhouette_samples(Xa, lab3) < 0
print(f"7  pixels with negative silhouette: {100 * neg.mean():.1f} %; next to a cluster boundary in the image: "
      f"{100 * edge_c[neg].mean():.0f} % (all pixels {100 * edge_c.mean():.0f} %)")

for name, f in [('(x - median)/std', lambda Z: (Z - np.median(Z, 0)) / Z.std(0)),
                ('(x - mean)/MAD', lambda Z: (Z - Z.mean(0)) / np.median(np.abs(Z - np.median(Z, 0)), 0)),
                ('(x - median)/(IQR/4)', lambda Z: (Z - np.median(Z, 0)) / (0.25 * np.subtract(*np.percentile(Z, [75, 25], 0))))]:
    l = KMeans(3, n_init=20, random_state=42).fit_predict(RobustScaler().fit_transform(1 / (1 + np.exp(-f(X_afm)))))
    print(f"8  sigmoid of {name:22s}: ARI vs untransformed K-Means k=3 = {adjusted_rand_score(lab3, l):.3f}")

sub = np.random.default_rng(0).choice(len(Xa), 5000, replace=False)
lh = HDBSCAN(min_cluster_size=100, min_samples=5).fit_predict(Xa[sub])
for c in np.unique(lh):
    m = lh == c
    print(f"9  HDBSCAN label {c:2d}: {100 * m.mean():4.1f} % of the subsample, scaffold {100 * y_afm[sub][m].mean():3.0f} %, "
          f"glass {100 * glass_afm[sub][m].mean():3.0f} %")


def wta_soft(lab):
    share = {c: y_afm[lab == c].mean() for c in np.unique(lab)}
    return (lab == max(share, key=share.get)).astype(int), np.isin(lab, [c for c, v in share.items() if v >= 0.5]).astype(int), share


same = []
for name, fit in algorithms.items():
    for k in range(2, 6):
        lab = fit(k); _, _, share = wta_soft(lab)
        same.append(max(share, key=share.get) == max(np.unique(lab), key=lambda c: X_afm[lab == c, 0].mean()))
print(f"10 highest-mean-H cluster equals the winner-take-all cluster in {sum(same)} of {len(same)} settings")


def ward_sub(k, n_sub, seed):
    idx = np.random.default_rng(seed).choice(len(Xa), n_sub, replace=False)
    lab = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Xa[idx])
    return NearestCentroid().fit(Xa[idx], lab).predict(Xa)


for k in (2, 3):
    rows = []
    for n_sub in (2000, 5000, 10000):
        for seed in (42, 0, 1, 2, 3):
            lab = ward_sub(k, n_sub, seed); wta, soft, _ = wta_soft(lab)
            rows.append((silhouette_score(Xa[sub], lab[sub]), matthews_corrcoef(y_afm, wta), matthews_corrcoef(y_afm, soft)))
    r = np.array(rows)
    print(f"11 Ward k={k}, subsample 2,000/5,000/10,000 x seeds 42,0,1,2,3: silhouette {r[:, 0].min():.3f}-{r[:, 0].max():.3f}, "
          f"MCC WTA {r[:, 1].min():.3f}-{r[:, 1].max():.3f}, MCC soft {r[:, 2].min():.3f}-{r[:, 2].max():.3f}")

lab = algorithms['GMM'](2)
extreme = (np.abs(Xa) > np.percentile(np.abs(Xa), 99, axis=0)).any(axis=1)
small = min((0, 1), key=lambda c: (lab == c).sum()); m = lab == small
print(f"12 GMM k=2 small cluster: {100 * m.mean():.1f} % of pixels, holds {100 * (glass_afm & m).sum() / glass_afm.sum():.0f} % of glass "
      f"and {100 * (extreme & m).sum() / extreme.sum():.0f} % of pixels beyond the 99th percentile in any channel")
