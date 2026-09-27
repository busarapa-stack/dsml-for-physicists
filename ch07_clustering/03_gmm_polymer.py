"""Ch.7  Gaussian mixture for toy polymer conformations (Rg, Ree); responsibilities and BIC.

Book references: sec:gmm, ssec:gmm-polymer, ssec:bic, code:gmm-polymer (line for line)
The listing uses numpy from the earlier listings (01_kmeans_elbow.py), which are executed first.
Data: SYNTHETIC three-state chain, seed 2026.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
Path('outputs').mkdir(exist_ok=True)
_src = (HERE / '01_kmeans_elbow.py').read_text()
exec(_src[_src.index('\n# --- code:'):_src.index('\n# --- end of listings')])   # earlier listings

# --- code:gmm-polymer --------------------------------------------------------
from sklearn.mixture import GaussianMixture

def make_polymer_states(n=3000, seed=2026):
    """Toy (Rg, Ree) samples of a chain in three metastable states."""
    rng = np.random.default_rng(seed)
    means = np.array([[2.2, 2.5], [3.5, 6.0], [5.0, 12.0]])   # folded, mid, open
    covs  = [np.array([[0.04, 0.03], [0.03, 0.30]]),
             np.array([[0.10, 0.20], [0.20, 1.20]]),
             np.array([[0.30, 0.90], [0.90, 4.00]])]
    pops  = np.array([0.5, 0.2, 0.3])                          # equilibrium weights
    state = rng.choice(3, size=n, p=pops)
    X = np.array([rng.multivariate_normal(means[s], covs[s]) for s in state])
    return X, state

X_polymer, state_true = make_polymer_states()
gmm = GaussianMixture(n_components=3, covariance_type='full',
                      n_init=10, random_state=42).fit(X_polymer)
labels = gmm.predict(X_polymer)
gamma = gmm.predict_proba(X_polymer)            # responsibilities
uncertain = gamma.max(axis=1) < 0.7             # candidate transition frames
print(gmm.weights_, gmm.means_, uncertain.sum())
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    names = ['folded', 'mid', 'open']
    print(f"uncertain frames: {uncertain.sum()} ({100 * uncertain.mean():.1f} %); "
          f"by true state {dict(zip(names, np.bincount(state_true[uncertain], minlength=3).tolist()))}")
    for k in range(3):
        w, v = np.linalg.eigh(gmm.covariances_[k])
        print(f"component {k}: mean {gmm.means_[k].round(2)}, principal eigenvector (Rg, Ree) = "
              f"{np.abs(v[:, -1]).round(3)}, eigenvalues {w.round(3)}")
    bic = [GaussianMixture(n_components=k, covariance_type='full', n_init=10,
                           random_state=42).fit(X_polymer).bic(X_polymer) for k in range(1, 7)]
    print("BIC for K = 1..6:", [round(b) for b in bic], "-> minimum at K =", 1 + int(np.argmin(bic)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(X_polymer[:, 0], X_polymer[:, 1], c=labels, s=5, cmap='viridis')
    axes[0].scatter(*X_polymer[uncertain].T, c='red', s=12, label='max gamma < 0.7')
    axes[0].set_xlabel('R_g'); axes[0].set_ylabel('R_ee'); axes[0].legend()
    axes[1].plot(range(1, 7), bic, 'o-'); axes[1].set_xlabel('K'); axes[1].set_ylabel('BIC')
    fig.tight_layout(); fig.savefig('outputs/ch07_gmm_polymer.png', dpi=100)
    print("saved outputs/ch07_gmm_polymer.png")
