"""Ch.8  Confidence interval of the mean of the first three eigenvalues of the Rouse trajectory:
independent-frame bootstrap vs block bootstrap.

Book references: ssec:dr-ci, E8.5, review question 7
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
_s = (HERE / '04_rouse_pca.py').read_text()
exec(_s[_s.index('import numpy'):_s.index('\n# --- end of listings')].replace("os.chdir(Path(__file__).resolve().parent)", ""))

SEED = 2026
B = 200


def mean_top3(Z):
    return PCA(n_components=3).fit(Z).explained_variance_.mean()


print(f"mean of the first three eigenvalues: {mean_top3(X_centered):.2f} (theory {1 / (12 * np.sin(np.pi / 100)**2):.2f})")
rng = np.random.default_rng(SEED)
iid = [mean_top3(X_centered[rng.integers(0, len(X_centered), len(X_centered))]) for _ in range(B)]
lo, hi = np.percentile(iid, [2.5, 97.5])
print(f"independent-frame bootstrap 95 % CI: [{lo:.1f}, {hi:.1f}]  width {hi - lo:.1f}")
for L in (100, 200, 500):
    nb = len(X_centered) // L
    rng = np.random.default_rng(SEED)
    blk = []
    for _ in range(B):
        start = rng.integers(0, nb, nb)
        blk.append(mean_top3(X_centered[(start[:, None] * L + np.arange(L)).ravel()]))
    blo, bhi = np.percentile(blk, [2.5, 97.5])
    print(f"block bootstrap, blocks of {L} frames: [{blo:.1f}, {bhi:.1f}]  width {bhi - blo:.1f} "
          f"({(bhi - blo) / (hi - lo):.1f} x wider)")
