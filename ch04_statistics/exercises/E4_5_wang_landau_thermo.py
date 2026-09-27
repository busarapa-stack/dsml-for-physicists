"""E4.5  Thermodynamics of the 4 x 4 Ising model from Wang-Landau g(E), against the EXACT
result from enumerating all 2^16 states.
"""
import itertools
import os
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
src = (HERE / '08_wang_landau_ising.py').read_text()
exec(src[src.index('rng = np.random.default_rng(seed=2026)'):src.index('E_bins, log_g, n_sweeps = wang_landau_2d_ising')])

E_bins, log_g, n_sweeps = wang_landau_2d_ising(L=4, ln_f_final=1e-6)
states = np.array(list(itertools.product([-1, 1], repeat=16))).reshape(-1, 4, 4)
E_all = -np.sum(states * (np.roll(states, 1, 1) + np.roll(states, 1, 2)), axis=(1, 2))
Ex, gx = np.unique(E_all, return_counts=True)

T = np.array([0.5, 1.0, 1.5, 2.0, 2.27, 2.5, 3.0, 4.0, 5.0])
wl, ex = thermodynamic(E_bins, log_g, T), thermodynamic(Ex, np.log(gx), T)
N = 16
print("   T    <E>/N WL  exact     C_v/N WL  exact     S/N WL  exact")
for i, t in enumerate(T):
    print(f"{t:5.2f}  {wl['E_mean'][i] / N:8.4f} {ex['E_mean'][i] / N:7.4f}   "
          f"{wl['Cv'][i] / N:8.4f} {ex['Cv'][i] / N:7.4f}   {wl['S'][i] / N:7.4f} {ex['S'][i] / N:7.4f}")
Tf = np.linspace(0.5, 5, 451)
kw, ke = np.argmax(thermodynamic(E_bins, log_g, Tf)['Cv']), np.argmax(thermodynamic(Ex, np.log(gx), Tf)['Cv'])
print(f"C_v peak: WL T = {Tf[kw]:.2f}, exact T = {Tf[ke]:.2f} (L = 4, far from T_c = 2.269: finite size)")
print(f"S/N -> ln 2 = {np.log(2):.4f} as T -> infinity (exact S/N at T = 5: {ex['S'][-1] / N:.4f})")
rel = lambda q: np.max(np.abs(np.array(wl[q]) / np.array(ex[q]) - 1)[1:])
print(f"largest relative difference WL vs exact for T >= 1: <E> {100 * rel('E_mean'):.1f} %, "
      f"C_v {100 * rel('Cv'):.1f} %, S {100 * rel('S'):.1f} %")
print("8 x 8 lattice and comparison with the exact finite-lattice result: see E4_5b_wang_landau_8x8.py")
