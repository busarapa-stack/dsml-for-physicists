"""E7.5  AFM protocol: choose k from internal indices only, then evaluate (F1, MCC, balanced accuracy, both rules),
interpret clusters in physical units, and compare feature subsets.  SYNTHETIC map from code:afm-3algo."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE); Path('outputs').mkdir(exist_ok=True)
for _f in ['01_kmeans_elbow.py', '03_gmm_polymer.py', '04_silhouette.py']:
    _s = (HERE / _f).read_text(); exec(_s[_s.index('\n# --- code:'):_s.index('\n# --- end of listings')])
_s = (HERE / '05_afm_three_algorithms.py').read_text()
_s = _s[_s.index('\n# --- code:afm'):_s.index('\n# --- end of listings')]
exec(_s[:_s.index('sub = np.random')])            # data, scaler and the three algorithms, without the printing loop

from sklearn.metrics import balanced_accuracy_score, calinski_harabasz_score, davies_bouldin_score

sub = np.random.default_rng(0).choice(len(Xa), 5000, replace=False)
rules = {}
print("step 1: internal indices only (reference map not used)")
choice = {}
for name, fit in algorithms.items():
    rows = []
    for k in range(2, 6):
        lab = fit(k)
        rows.append((k, silhouette_score(Xa[sub], lab[sub]), davies_bouldin_score(Xa, lab), calinski_harabasz_score(Xa, lab), lab))
    for k, s, d, c, _ in rows:
        print(f"  {name:8s} k={k}: silhouette {s:.3f}  DB {d:.3f}  CH {c:.0f}")
    k_sil = max(rows, key=lambda r: r[1])[0]; k_db = min(rows, key=lambda r: r[2])[0]; k_ch = max(rows, key=lambda r: r[3])[0]
    print(f"  {name}: silhouette -> k={k_sil}, Davies-Bouldin -> k={k_db}, Calinski-Harabasz -> k={k_ch}; chosen k = {k_sil} (silhouette)")
    choice[name] = [r for r in rows if r[0] == k_sil][0][4]


def to_binary(lab):
    share = {c: y_afm[lab == c].mean() for c in np.unique(lab)}
    wta = (lab == max(share, key=share.get)).astype(int)
    soft = np.isin(lab, [c for c, v in share.items() if v >= 0.5]).astype(int)
    return wta, soft


print("\nstep 2: evaluation of the chosen k against the reference map")
base = np.ones_like(y_afm)
print(f"  all-scaffold baseline: F1 {f1_score(y_afm, base):.3f}  MCC {matthews_corrcoef(y_afm, base):.3f}  "
      f"balanced acc {balanced_accuracy_score(y_afm, base):.3f}")
for name, lab in choice.items():
    for rule, pred in zip(['WTA', 'soft'], to_binary(lab)):
        print(f"  {name:8s} {rule:4s}: F1 {f1_score(y_afm, pred):.3f}  MCC {matthews_corrcoef(y_afm, pred):.3f}  "
              f"balanced acc {balanced_accuracy_score(y_afm, pred):.3f}")

print("\nstep 3: K-Means k=3 clusters in physical units (median H nm, S N/m, A nN)")
lab = choice['K-Means']
for c in np.unique(lab):
    m = lab == c
    print(f"  cluster {c}: {100 * m.mean():4.1f} %  scaffold {100 * y_afm[m].mean():4.1f} %  glass {100 * glass_afm[m].mean():4.1f} %  "
          f"median {np.median(X_afm[m], axis=0).round(2)}")

print("\nstep 4: feature subsets (K-Means, k = 3, RobustScaler on the subset)")
for fs in ['H', 'S', 'A', 'HA', 'HSA']:
    Z = RobustScaler().fit_transform(X_afm[:, ['HSA'.index(c) for c in fs]])
    lab = KMeans(3, n_init=20, random_state=42).fit_predict(Z)
    wta, soft = to_binary(lab)
    print(f"  {fs:3s}: MCC WTA {matthews_corrcoef(y_afm, wta):.3f}  soft {matthews_corrcoef(y_afm, soft):.3f}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
imgs = [(y_afm + 0.5 * glass_afm, 'reference')] + [(algorithms[n](k), f'{n} k={k}') for k in (2, 3) for n in algorithms]
for ax, (img, t) in zip(axes.ravel(), imgs):
    ax.imshow(np.asarray(img, float).reshape(128, 128)); ax.set_title(t); ax.axis('off')
axes.ravel()[-1].axis('off')
fig.tight_layout(); fig.savefig('outputs/E7_5_afm_maps.png', dpi=80)
print("saved outputs/E7_5_afm_maps.png")
