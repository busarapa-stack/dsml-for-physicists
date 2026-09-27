"""Ch.10  Lennard-Jones chain (N = 30): Monte Carlo ensemble at 15 temperatures and physical indicators.

Book references: ssec:polymer-data, ssec:polymer-fe, code:chain-gen and code:chain-features (line for line)
Writes data/chain.npz (conf, ener, T_lab, T_grid), reused by 04, 06 and exercise E10.6.
Numbers quoted in the text: generation about two minutes; <Rg^2> 2.6 (T = 0.6) -> 8.6 (T = 3.4);
tanh fit of <Rg^2>(T): T* = 1.90 +/- 0.04, w = 1.0; d<Rg^2>/dT 2.6-3.1 for T = 1.4-2.2, no sharp peak;
C_v/N decreasing with a weak shoulder, large at T = 0.6.
"""
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
Path('data').mkdir(exist_ok=True)
Path('outputs').mkdir(exist_ok=True)
_t0 = time.time()

# --- code:chain-gen ----------------------------------------------------------
import numpy as np

def make_chain_ensemble(N=30, T=2.0, n_samples=400, sweeps_between=10,
                        n_equil=2000, seed=0):
    """Flexible chain: bond length 1, Lennard-Jones (eps = sigma = 1) between
    beads with |i - j| >= 2. Metropolis sampling with pivot and crankshaft moves.
    Returns centred configurations (n_samples, N, 3) and energies."""
    rng = np.random.default_rng(seed)
    iu, ju = np.triu_indices(N, k=2)
    def energy(r):
        s6 = 1.0 / np.sum((r[iu] - r[ju])**2, axis=1)**3
        return np.sum(4.0 * (s6 * s6 - s6))
    def rotation(axis, angle):                         # Rodrigues' formula
        a = axis / np.linalg.norm(axis)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    r = np.zeros((N, 3)); r[:, 0] = np.arange(N)       # straight initial chain
    E, beta = energy(r), 1.0 / T
    confs, energies = [], []
    for m in range((n_equil + n_samples * sweeps_between) * N):
        new = r.copy()
        if rng.random() < 0.5:                         # pivot move
            k = rng.integers(1, N - 1)
            R = rotation(rng.normal(size=3), rng.uniform(-np.pi, np.pi) * 0.5)
            if rng.random() < 0.5:
                new[k+1:] = (r[k+1:] - r[k]) @ R.T + r[k]
            else:
                new[:k] = (r[:k] - r[k]) @ R.T + r[k]
        else:                                          # crankshaft or end move
            i = rng.integers(0, N)
            if 0 < i < N - 1:
                R = rotation(r[i+1] - r[i-1], rng.uniform(-np.pi, np.pi))
                new[i] = (r[i] - r[i-1]) @ R.T + r[i-1]
            else:
                j = 1 if i == 0 else N - 2
                v = rng.normal(size=3)
                new[i] = r[j] + v / np.linalg.norm(v)
        E_new = energy(new)
        if E_new <= E or rng.random() < np.exp(-beta * (E_new - E)):
            r, E = new, E_new
        sweep, rest = divmod(m + 1, N)
        if rest == 0 and sweep > n_equil and (sweep - n_equil) % sweeps_between == 0:
            confs.append(r - r.mean(axis=0))
            energies.append(E)
    return np.array(confs), np.array(energies)

T_grid = np.round(np.arange(0.6, 3.41, 0.2), 1)       # 15 temperatures
data = [make_chain_ensemble(T=T, seed=100 + k) for k, T in enumerate(T_grid)]
conf = np.concatenate([d[0] for d in data])            # (6000, 30, 3)
ener = np.concatenate([d[1] for d in data])
T_lab = np.repeat(T_grid, 400)
# --- code:chain-features -----------------------------------------------------
N = conf.shape[1]
iu, ju = np.triu_indices(N, k=1)
dist = np.linalg.norm(conf[:, iu] - conf[:, ju], axis=2)   # 435 pair distances
rg2 = np.mean(np.sum(conf**2, axis=2), axis=1)              # squared radius of gyration
for T in T_grid:
    m = T_lab == T
    dRg2_dT = np.cov(rg2[m], ener[m])[0, 1] / T**2           # fluctuation formula
    print(f"T = {T:.1f}  <Rg^2> = {rg2[m].mean():5.2f}  "
          f"C_v/N = {ener[m].var() / T**2 / N:5.3f}  d<Rg^2>/dT = {dRg2_dT:4.2f}")
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    print(f"generation + features: {time.time() - _t0:.0f} s")
    np.savez('data/chain.npz', conf=conf, ener=ener, T_lab=T_lab, T_grid=T_grid)
    print("saved data/chain.npz", conf.shape)

    from scipy.optimize import curve_fit
    mean_rg2 = np.array([rg2[T_lab == T].mean() for T in T_grid])
    f = lambda T, a, b, Ts, w: a + b * np.tanh((T - Ts) / w)
    p, c = curve_fit(f, T_grid, mean_rg2, p0=[5, 3, 1.8, 0.5])
    e = np.sqrt(np.diag(c))
    print(f"tanh fit of <Rg^2>(T): T* = {p[2]:.2f} +/- {e[2]:.2f}, w = {p[3]:.2f} +/- {e[3]:.2f}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(T_grid, mean_rg2, 'o', label=r'$\langle R_g^2\rangle$')
    Ts = np.linspace(0.6, 3.4, 200)
    ax.plot(Ts, f(Ts, *p), '-', label=f'tanh fit, T* = {p[2]:.2f}')
    ax.set_xlabel('T'); ax.set_ylabel(r'$\langle R_g^2\rangle$'); ax.legend()
    fig.tight_layout(); fig.savefig('outputs/ch10_chain_rg2.png', dpi=100)
    print("saved outputs/ch10_chain_rg2.png")
