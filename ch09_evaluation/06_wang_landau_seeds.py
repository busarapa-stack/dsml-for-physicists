"""Ch.9  Run-to-run spread of Wang-Landau for the 4x4 Ising model: ten independent runs (seeds 0-9, ln f_final = 1e-6)
compared with the exact C_v/N peak from enumerating all 2^16 states.

Book references: ssec:wl-secondary; code:wl of Chapter 4 (executed from ../ch04_statistics).  About 4 minutes on one core
(runs in parallel when several cores are available).
"""
import itertools
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
_s = (HERE.parent / 'ch04_statistics' / '08_wang_landau_ising.py').read_text()
exec(_s[_s.index('# from code:metropolis'):_s.index('E_bins, log_g, n_sweeps = wang_landau_2d_ising')])
T_GRID = np.arange(1.5, 3.5, 0.001)


def cv_peak(E, lg):
    cvs = np.array(thermodynamic(E, lg, T_GRID)['Cv']) / 16
    k = cvs.argmax()
    return cvs[k], T_GRID[k]


def one_run(seed):
    global rng
    rng = np.random.default_rng(seed)                 # the listing draws from the global rng
    E, lg, sweeps = wang_landau_2d_ising(L=4, ln_f_final=1e-6)
    return seed, cv_peak(E, lg), sweeps


if __name__ == '__main__':
    counts = {}
    for bits in itertools.product([-1, 1], repeat=16):
        E = total_energy(np.array(bits).reshape(4, 4))
        counts[E] = counts.get(E, 0) + 1
    Ex = np.array(sorted(counts))
    peak_x, T_x = cv_peak(Ex, np.log([counts[k] for k in Ex]))
    print(f"exact: C_v/N peak {peak_x:.4f} at T = {T_x:.3f}")
    with Pool(min(10, os.cpu_count() or 1)) as pool:
        res = pool.map(one_run, range(10))
    pk = np.array([r[1][0] for r in res]); Tp = np.array([r[1][1] for r in res])
    for s, (p, Tpk), sw in res:
        print(f"  seed {s}: peak {p:.4f} at T = {Tpk:.3f} ({sw} sweeps)")
    print(f"single runs {pk.min():.3f}-{pk.max():.3f}; mean {pk.mean():.4f} +/- {pk.std(ddof=1) / np.sqrt(10):.4f} "
          f"(sd {pk.std(ddof=1):.4f}); peak T {Tp.mean():.3f} +/- {Tp.std(ddof=1) / np.sqrt(10):.3f}")
