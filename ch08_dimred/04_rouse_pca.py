"""Ch.8  Rouse chain (exact normal-mode sampling) and PCA of the Cartesian trajectory.

Book references: sec:polymer-primary, code:rouse-gen, code:polymer-pca (line for line), ssec:rouse-modes
numpy and PCA are imported in code:pca (01_pca_bootstrap.py); they are imported here directly.
Units k_B T = friction = bond length = 1.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

os.chdir(Path(__file__).resolve().parent)
Path('outputs').mkdir(exist_ok=True)

# --- code:rouse-gen ----------------------------------------------------------
def make_rouse_trajectory(N=50, n_frames=10000, dt_frame=5.0, seed=2026):
    """Rouse chain (k_B T = friction = bond length = 1) sampled exactly in
    its normal-mode basis; returns Cartesian coordinates (n_frames, 3N)."""
    rng = np.random.default_rng(seed)
    k = 3.0                                              # 3 k_B T / b^2
    i = np.arange(N)
    p = np.arange(1, N)                                  # p = 0 (centre of mass) removed
    Q = np.sqrt(2.0 / N) * np.cos(np.outer(p, np.pi * (i + 0.5) / N))
    a = 4.0 * np.sin(p * np.pi / (2 * N))**2             # Rouse-matrix eigenvalues
    var = 1.0 / (k * a)                                  # <X_p^2> per axis
    tau = 1.0 / (k * a)                                  # relaxation times
    decay = np.exp(-dt_frame / tau)
    Xp = np.empty((n_frames, len(p), 3))
    Xp[0] = rng.normal(size=(len(p), 3)) * np.sqrt(var)[:, None]
    for t in range(1, n_frames):                         # exact Ornstein-Uhlenbeck step
        Xp[t] = (decay[:, None] * Xp[t - 1]
                 + np.sqrt(var * (1 - decay**2))[:, None] * rng.normal(size=(len(p), 3)))
    r = np.einsum('pi,tpc->tic', Q, Xp)                 # bead positions (CM at origin)
    return r.reshape(n_frames, 3 * N), tau

polymer_trajectory, tau_theory = make_rouse_trajectory()
# --- code:polymer-pca --------------------------------------------------------
X = polymer_trajectory                           # (10000, 150), same units in every column
X_centered = X - X.mean(axis=0)                  # no standardisation (see text)

pca = PCA(n_components=12)
X_pca = pca.fit_transform(X_centered)
print("variance ratio:", np.round(pca.explained_variance_ratio_[:9], 3))
print("eigenvalues:  ", np.round(pca.explained_variance_[:9], 1))

N = 50
mode1 = pca.components_[0].reshape(N, 3)         # PC1 as a displacement field
profile = mode1 @ np.linalg.svd(mode1)[2][0]     # project on its dominant direction
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    p = np.arange(1, N)
    a = 4 * np.sin(p * np.pi / (2 * N))**2
    lam_th = 1 / (3 * a)
    print(f"\ntau_1 = {tau_theory[0]:.1f}; record length {10000 * 5.0:.0f} = {10000 * 5.0 / tau_theory[0]:.0f} tau_1; "
          f"tau_1 = {tau_theory[0] / 5.0:.0f} frames")
    ev = pca.explained_variance_
    for q in range(3):
        print(f"group p={q + 1}: eigenvalues {ev[3*q:3*q+3].round(1)}, mean {ev[3*q:3*q+3].mean():.1f}, theory 1/(3 a_p) = {lam_th[q]:.1f}")
    r = pca.explained_variance_ratio_
    print(f"first 3 components {r[:3].sum():.3f} (theory {(1/a[0]) / (1/a).sum():.3f}; 6/pi^2 = {6/np.pi**2:.3f}); "
          f"first 9 {r[:9].sum():.3f} (theory {(1/a[:3]).sum() / (1/a).sum():.3f})")
    i = np.arange(N)
    print(f"|corr| PC1 profile vs cos(pi(i+1/2)/N) = {abs(np.corrcoef(profile, np.cos(np.pi * (i + .5) / N))[0, 1]):.4f}")
    for j in range(3, 6):
        m = pca.components_[j].reshape(N, 3); pr = m @ np.linalg.svd(m)[2][0]
        print(f"|corr| PC{j + 1} profile vs cos(2 pi(i+1/2)/N) = {abs(np.corrcoef(pr, np.cos(2 * np.pi * (i + .5) / N))[0, 1]):.4f}")
    z = X_pca[:, 0] - X_pca[:, 0].mean()
    acf = np.array([1.0] + [np.mean(z[:-l] * z[l:]) / np.mean(z * z) for l in range(1, 61)])
    for last in (10, 20, 40):                                   # fit ln C(t) on lags 1..last-1 frames
        lag = np.arange(1, last)
        slope = np.polyfit(lag * 5.0, np.log(acf[1:last]), 1)[0]
        print(f"tau_1 from ln C(t), lags up to {5 * (last - 1):.0f} time units: {-1 / slope:.1f} (theory {tau_theory[0]:.1f})")
    tot = X_centered.var(axis=0, ddof=1).sum()
    rg2 = (X_centered.reshape(-1, N, 3)**2).sum(axis=2).mean(axis=1).mean()
    print(f"sum of all variances {tot:.1f} = N <Rg^2> -> <Rg^2> = {tot / N:.2f} (direct {rg2:.2f}); "
          f"Gaussian chain N b^2/6 = {N / 6:.2f}, exact Rouse (N^2-1)/(6N) = {(N**2 - 1) / (6 * N):.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    full = PCA().fit(X_centered).explained_variance_
    axes[0].semilogy(np.arange(1, 31), full[:30], 'o', label='PCA'); axes[0].semilogy(np.arange(1, 31), np.repeat(lam_th[:10], 3), 'k_', ms=12, label='1/(3 a_p)'); axes[0].legend()
    axes[0].set_xlabel('component'); axes[0].set_ylabel('eigenvalue'); axes[0].set_title('triplets, ~1/p^2')
    sgn = np.sign(np.corrcoef(profile, np.cos(np.pi * (i + .5) / N))[0, 1])      # the sign of a PC is arbitrary
    axes[1].plot(sgn * profile / np.abs(profile).max(), label='PC1 (sign aligned)'); axes[1].plot(np.cos(np.pi * (i + .5) / N), '--', label='cos p=1')
    axes[1].legend(); axes[1].set_xlabel('bead')
    axes[2].plot(np.arange(61) * 5, acf); axes[2].plot(np.arange(61) * 5, np.exp(-np.arange(61) * 5 / tau_theory[0]), '--')
    axes[2].set_xlabel('t'); axes[2].set_title('C(t) of PC1 vs exp(-t/tau_1)')
    fig.tight_layout(); fig.savefig('outputs/ch08_rouse_pca.png', dpi=90)
    print("saved outputs/ch08_rouse_pca.png")
