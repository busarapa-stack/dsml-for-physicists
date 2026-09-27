"""Ch.9  Posterior of the covariance of the Rouse trajectory (inverse-Wishart with N_eff): eigenvalue and eigenvector uncertainty.

Book references: ssec:cross-case, code:bayes-cov (line for line)
polymer_trajectory comes from code:rouse-gen (Chapter 8, executed from ../ch08_dimred);
tau_int from code:mc-tau (05_mc_tau.py, function definition only).
"""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
_s = (HERE.parent / 'ch08_dimred' / '04_rouse_pca.py').read_text()
exec(_s[_s.index('\n# --- code:rouse-gen'):_s.index('\n# --- code:polymer-pca')])   # polymer_trajectory
_s = (HERE / '05_mc_tau.py').read_text()
exec(_s[_s.index('def tau_int'):_s.index('def block_bootstrap')])                     # tau_int only

# --- code:bayes-cov ----------------------------------------------------------
from scipy.stats import invwishart
from sklearn.decomposition import PCA

X = polymer_trajectory - polymer_trajectory.mean(axis=0)   # Chapter 8 data
n, D = X.shape
S = X.T @ X / n                                   # sample covariance
pc1 = PCA(n_components=1).fit_transform(X)[:, 0]
tau_f, _ = tau_int(pc1)                           # in frames
n_eff = n / (2 * tau_f)                           # effectively independent frames
nu0, Psi0 = D + 2, 1e-3 * np.eye(D)               # weak, proper prior
post = invwishart(df=nu0 + n_eff, scale=Psi0 + n_eff * S)

N = 50
q1 = np.sqrt(2 / N) * np.cos(np.pi * (np.arange(N) + 0.5) / N)   # Rouse p = 1
P1 = np.kron(q1[:, None], np.eye(3))              # x, y, z copies, shape (150, 3)
rng = np.random.default_rng(seed=2026)
lam1, overlap = [], []
for Sig in post.rvs(size=1000, random_state=rng):
    w, V = np.linalg.eigh(Sig)
    lam1.append(w[-3:].mean())                    # degenerate p = 1 triplet
    overlap.append(np.sum((P1.T @ V[:, -3:])**2) / 3)
print("n_eff =", round(n_eff))
print("lambda_1 95% interval:", np.percentile(lam1, [2.5, 50, 97.5]).round(1))
print("overlap with p=1 mode:", np.percentile(overlap, [2.5, 50, 97.5]).round(3))
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    print(f"tau of PC1 = {tau_f:.1f} frames (theory 84.5 / 5 = {84.46 / 5:.1f})")
