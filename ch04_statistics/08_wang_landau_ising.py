"""Ch.4  Wang-Landau density of states for the 2D Ising model (L = 4), with checks.

Book references: ssec:wang-landau, def:wl, code:wl, tab:wl-thermo, exercise E4.5
The listing is reproduced line for line. Checks added:
  * sum of g(E) against 2^16 for seeds 1-8, at ln f = 1e-6 and 1e-8 (error saturation)
  * g(E) against the exact values from enumerating all 65,536 states
  * entropy per spin -> ln 2 at high T
"""
import itertools
import os
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).resolve().parent)
rng = np.random.default_rng(seed=2026)

# from code:metropolis
def total_energy(lattice):
    # each bond counted once: one neighbour along each axis
    return -np.sum(lattice * (np.roll(lattice, 1, 0) + np.roll(lattice, 1, 1)))

# --- code:wl ----------------------------------------------------------------
def wang_landau_2d_ising(L, ln_f_final=1e-8, flatness=0.8,
                         check_every=1000, max_sweeps=10**7):
    lattice = rng.choice([-1, 1], size=(L, L))
    N = L * L
    E_min = -2 * N
    E_bins = np.arange(E_min, 2 * N + 1, 4)
    log_g = np.zeros(len(E_bins))
    H = np.zeros(len(E_bins), dtype=int)
    visited = np.zeros(len(E_bins), dtype=bool)   # some bins never occur
    ln_f = 1.0
    idx = lambda E: (E - E_min) // 4

    E_curr = total_energy(lattice)
    sweep = 0
    while ln_f > ln_f_final and sweep < max_sweeps:
        for _ in range(N):
            i, j = rng.integers(0, L, size=2)
            s = lattice[i, j]
            nb = (lattice[(i+1) % L, j] + lattice[(i-1) % L, j]
                + lattice[i, (j+1) % L] + lattice[i, (j-1) % L])
            E_new = E_curr + 2 * s * nb
            if rng.random() < np.exp(log_g[idx(E_curr)] - log_g[idx(E_new)]):
                lattice[i, j] = -s
                E_curr = E_new
            k = idx(E_curr)
            log_g[k] += ln_f
            H[k] += 1
            visited[k] = True
        sweep += 1
        if sweep % check_every == 0:
            h = H[visited]
            if h.min() >= flatness * h.mean():    # histogram is flat enough
                ln_f /= 2.0
                H[:] = 0
    E_bins, log_g = E_bins[visited], log_g[visited]
    log_g += np.log(2.0) - log_g[0]               # anchor: g(E_min) = 2
    return E_bins, log_g, sweep

def thermodynamic(E_bins, log_g, T_values):
    res = {'T': T_values, 'E_mean': [], 'Cv': [], 'S': []}
    for T in T_values:
        beta = 1.0 / T
        log_w = log_g - beta * E_bins
        log_w_max = log_w.max()                   # log-sum-exp trick
        w = np.exp(log_w - log_w_max)
        Z = w.sum()
        E_mean = (E_bins * w).sum() / Z
        E2_mean = (E_bins**2 * w).sum() / Z
        res['E_mean'].append(E_mean)
        res['Cv'].append(beta**2 * (E2_mean - E_mean**2))
        res['S'].append(np.log(Z) + log_w_max + beta * E_mean)   # S/k_B
    return res

E_bins, log_g, n_sweeps = wang_landau_2d_ising(L=4, ln_f_final=1e-6)
print(np.exp(log_g).sum(), 2**16)               # sanity check

# --- exact density of states by enumeration ------------------------------------
states = np.array(list(itertools.product([-1, 1], repeat=16))).reshape(-1, 4, 4)
E_all = -np.sum(states * (np.roll(states, 1, 1) + np.roll(states, 1, 2)), axis=(1, 2))
vals, cnt = np.unique(E_all, return_counts=True)
exact = dict(zip(vals.tolist(), cnt.tolist()))
print("exact g(E):", exact)
print("no states at E = -28 or +28:", -28 not in exact and 28 not in exact)
print(f"this run (seed 2026): sum g = {np.exp(log_g).sum():.0f} "
      f"({100 * (np.exp(log_g).sum() / 2**16 - 1):+.1f} %), {n_sweeps} sweeps")

# --- several seeds and two final ln f -------------------------------------------
print("\nseed  ln_f=1e-6: sum error  g(-24)   g(-20) | ln_f=1e-8: sum error")
for seed in range(1, 9):
    out = []
    for lnf in (1e-6, 1e-8):
        rng = np.random.default_rng(seed)
        Eb, lg, _ = wang_landau_2d_ising(L=4, ln_f_final=lnf)
        out.append((Eb, np.exp(lg)))
    (Eb, g6), (_, g8) = out
    d = dict(zip(Eb.tolist(), g6))
    print(f"{seed:4d}       {100 * (g6.sum() / 2**16 - 1):+6.1f} %  {d[-24]:6.1f}  {d[-20]:6.1f}"
          f"  |        {100 * (g8.sum() / 2**16 - 1):+6.1f} %")
print("-> lowering ln f from 1e-6 to 1e-8 changes almost nothing: error saturation")

# where does the error sit? relative error of g(E) along the energy axis (seed 3)
rng = np.random.default_rng(3)
Eb, lg, _ = wang_landau_2d_ising(L=4, ln_f_final=1e-6)
print("\nseed 3, error of g(E) vs exact, from the anchor E_min outward:")
print("  " + "  ".join(f"{e}:{100 * (np.exp(l) / exact[e] - 1):+.0f}%" for e, l in zip(Eb, lg)))
print("-> small next to the anchor g(-32) = 2, larger far from it; g(E) and g(-E) can differ"
      " by tens of percent without any bug")

# --- thermodynamics from g(E) of the seed-2026 run (E_bins, log_g of code:wl) ------
T_values = np.linspace(0.5, 5.0, 91)
res = thermodynamic(E_bins, log_g, T_values)
N = 16
k = int(np.argmax(res['Cv']))
print(f"\nC_v/N peak at T = {T_values[k]:.2f} (L = 4); S/N at T = 5: {res['S'][-1] / N:.3f}, "
      f"ln 2 = {np.log(2):.3f}; S/N at T = 50: {thermodynamic(E_bins, log_g, [50.0])['S'][0] / N:.3f}")
print("(at infinite T, S/N = ln(sum g)/N; it misses ln 2 by exactly the error of the sum)")
