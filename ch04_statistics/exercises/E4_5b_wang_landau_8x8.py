"""E4.5 (second part)  Wang-Landau for the 8 x 8 Ising model with ln f = 1e-8.

Checks three statements of the book on the 8 x 8 lattice:
  * how long code:wl takes in pure Python (compared with 4 x 4 at the same ln f),
  * whether <E>/N and C_v/N agree with the Metropolis L = 8 scan of E4.4,
  * how close the result is to the EXACT answer.

The exact reference is Kaufman's closed-form partition function of the finite
L x L lattice with periodic boundaries (B. Kaufman, Phys. Rev. 76, 1232 (1949);
A. E. Ferdinand and M. E. Fisher, Phys. Rev. 185, 832 (1969)).  The function
lnZ_kaufman() is checked below against full enumeration of the 4 x 4 lattice.
"""
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
src = (HERE / '08_wang_landau_ising.py').read_text()
exec(src[src.index('rng = np.random.default_rng(seed=2026)'):src.index('E_bins, log_g, n_sweeps = wang_landau_2d_ising')])


def lnZ_kaufman(L, T):
    """Exact ln Z of the L x L periodic 2D Ising model (J = k_B = 1)."""
    K = 1.0 / T
    l = np.arange(2 * L)
    gam = np.arccosh(np.cosh(2 * K) / np.tanh(2 * K) - np.cos(np.pi * l / L))
    gam[0] = 2 * K + np.log(np.tanh(K))            # gamma_0 keeps its sign
    terms = []
    for g_set in (gam[1::2], gam[0::2]):
        for f in (np.cosh, np.sinh):
            v = 2 * f(L * g_set / 2)
            terms.append((np.sum(np.log(np.abs(v))), np.prod(np.sign(v))))
    top = max(t[0] for t in terms)
    s = sum(sg * np.exp(lg - top) for lg, sg in terms)
    return np.log(0.5) + (L * L / 2) * np.log(2 * np.sinh(2 * K)) + top + np.log(s)


def exact_E_Cv(L, T, h=1e-4):
    """<E>/N and C_v/N from derivatives of ln Z with respect to beta."""
    b = 1.0 / T
    f = lambda bb: lnZ_kaufman(L, 1.0 / bb)
    E = -(f(b + h) - f(b - h)) / (2 * h)
    C = b**2 * (f(b + h) - 2 * f(b) + f(b - h)) / h**2
    return E / (L * L), C / (L * L)


# --- 1. check the exact formula against enumeration of all 2^16 states of 4 x 4 --------
states = np.array(list(itertools.product([-1, 1], repeat=16))).reshape(-1, 4, 4)
E_all = -np.sum(states * (np.roll(states, 1, 1) + np.roll(states, 1, 2)), axis=(1, 2))
Ex, gx = np.unique(E_all, return_counts=True)
worst = max(abs(np.log(np.sum(gx * np.exp(-Ex / T))) - lnZ_kaufman(4, T)) for T in (1.0, 2.27, 5.0))
print(f"Kaufman ln Z vs enumeration (4 x 4): largest difference {worst:.1e}")

# --- 2. timing: 4 x 4 and 8 x 8 at ln f = 1e-8 -----------------------------------------
t0 = time.time()
E4, lg4, n4 = wang_landau_2d_ising(L=4, ln_f_final=1e-8)
t4 = time.time() - t0
t0 = time.time()
E8, lg8, n8 = wang_landau_2d_ising(L=8, ln_f_final=1e-8)
t8 = time.time() - t0
print(f"4 x 4: {n4:7d} sweeps = {16 * n4:10,d} spin-flip attempts, {t4:5.1f} s")
print(f"8 x 8: {n8:7d} sweeps = {64 * n8:10,d} spin-flip attempts, {t8:5.1f} s")
print(f"-> 4 times more spins, {64 * n8 / (16 * n4):.0f} times more attempts "
      f"({len(E8)} instead of {len(E4)} energy levels)")
sum_err = np.exp(np.logaddexp.reduce(lg8) - 64 * np.log(2)) - 1
print(f"8 x 8: sum of g(E) differs from 2^64 by {100 * sum_err:+.1f} %")

# --- 3. thermodynamics: Wang-Landau, Metropolis (E4.4), exact ---------------------------
T = np.array([1.5, 2.0, 2.3, 2.4, 2.5, 3.0, 3.5])
wl = thermodynamic(E8, lg8, T)
ex = np.array([exact_E_Cv(8, t) for t in T])
met_file = Path('outputs/ch04_ising_metropolis_L8.csv')
met = pd.read_csv(met_file).set_index('T') if met_file.exists() else None
print("\n   T   <E>/N: WL    Metro   exact   |  C_v/N: WL   Metro   exact")
for i, t in enumerate(T):
    mE = mC = float('nan')
    if met is not None:
        row = met.loc[np.isclose(met.index, t)]
        mE, mC = row['E_per_spin'].iloc[0], row['Cv_per_spin'].iloc[0]
    print(f"{t:5.2f}  {wl['E_mean'][i] / 64:8.4f} {mE:8.4f} {ex[i, 0]:7.4f}   |  "
          f"{wl['Cv'][i] / 64:7.4f} {mC:7.4f} {ex[i, 1]:7.4f}")
if met is None:
    print("(run E4_4_metropolis_thermalization.py first to fill the Metropolis column)")

Tf = np.linspace(1.8, 3.0, 1201)
k_wl = np.argmax(thermodynamic(E8, lg8, Tf)['Cv'])
C_ex = np.array([exact_E_Cv(8, t)[1] for t in Tf])
k_ex = np.argmax(C_ex)
print(f"\nC_v/N peak (L = 8): WL T = {Tf[k_wl]:.3f} (height {thermodynamic(E8, lg8, [Tf[k_wl]])['Cv'][0] / 64:.3f}), "
      f"exact T = {Tf[k_ex]:.3f} (height {C_ex[k_ex]:.3f}); T_c(infinite) = 2.269")
rel = lambda q, j: np.max(np.abs(np.array(wl[q]) / 64 / ex[:, j] - 1))
print(f"largest relative difference WL vs exact on this grid: <E> {100 * rel('E_mean', 0):.1f} %, "
      f"C_v {100 * rel('Cv', 1):.1f} %")
