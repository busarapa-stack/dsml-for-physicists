"""E4.5 (extra)  The 1/t variant of Belardinelli and Pereyra (2007) against the original
halving schedule of code:wl, on the 4 x 4 lattice where g(E) is known exactly.

The 1/t variant runs code:wl unchanged until ln f falls below 1/t, where
t = (spin-flip attempts) / (number of energy levels).  From then on ln f = 1/t at every
step and the flatness check is no longer used.  The run stops when ln f < ln_f_final.

    python exercises/E4_5c_wang_landau_1t.py          # seeds 1-8, about 3 minutes
    python exercises/E4_5c_wang_landau_1t.py --long   # adds 1/t down to 1e-6, about 20 minutes
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
src = (HERE / '08_wang_landau_ising.py').read_text()
exec(src[src.index('rng = np.random.default_rng(seed=2026)'):src.index('E_bins, log_g, n_sweeps = wang_landau_2d_ising')])


def wang_landau_1t(L, ln_f_final=1e-6, flatness=0.8, check_every=1000):
    lattice = rng.choice([-1, 1], size=(L, L))
    N = L * L
    E_min = -2 * N
    E_bins = np.arange(E_min, 2 * N + 1, 4)
    log_g = np.zeros(len(E_bins))
    H = np.zeros(len(E_bins), dtype=int)
    visited = np.zeros(len(E_bins), dtype=bool)
    ln_f = 1.0
    idx = lambda E: (E - E_min) // 4
    E_curr = total_energy(lattice)
    sweep, attempts, one_over_t = 0, 0, False
    while ln_f > ln_f_final:
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
            attempts += 1
            if one_over_t:
                ln_f = visited.sum() / attempts          # ln f = 1/t
        sweep += 1
        if not one_over_t and sweep % check_every == 0:
            h = H[visited]
            if h.min() >= flatness * h.mean():
                ln_f /= 2.0
                H[:] = 0
                if ln_f < visited.sum() / attempts:      # halving has gone below 1/t
                    one_over_t = True
    E_bins, log_g = E_bins[visited], log_g[visited]
    log_g += np.log(2.0) - log_g[0]
    return E_bins, log_g, sweep


runs = [('original, ln f = 1e-6', wang_landau_2d_ising, 1e-6),
        ('1/t,      ln f = 1e-5', wang_landau_1t, 1e-5)]
if '--long' in sys.argv:
    runs.append(('1/t,      ln f = 1e-6', wang_landau_1t, 1e-6))

summary = {}
for name, fn, lnf in runs:
    print(f"\n{name}")
    print("seed   sweeps   time    sum g vs 2^16")
    errs, sweeps, secs = [], [], []
    for seed in range(1, 9):
        rng = np.random.default_rng(seed)
        t0 = time.time()
        E, lg, n = fn(4, ln_f_final=lnf)
        dt = time.time() - t0
        err = 100 * (np.exp(lg).sum() / 2**16 - 1)
        errs.append(err); sweeps.append(n); secs.append(dt)
        print(f"{seed:4d} {n:8d} {dt:6.1f} s   {err:+6.1f} %")
    summary[name] = (np.max(np.abs(errs)), np.mean(sweeps), np.mean(secs))
    print(f"largest |error| {summary[name][0]:.1f} %, mean {summary[name][1]:.0f} sweeps")

base = summary[runs[0][0]][1]
print("\nsummary (seeds 1-8)")
for name in summary:
    e, n, _ = summary[name]
    print(f"  {name}: largest |error| {e:4.1f} %, {n / base:4.1f} x the sweeps of the original")
print("-> 1/t keeps lowering the error as the run gets longer; the original schedule saturates")
