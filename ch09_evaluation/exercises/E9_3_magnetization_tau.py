"""E9.3  tau_int of e and |m| at T = 2.3 (same seed as code:mc-tau), and the SE of <|m|> by four methods.  About 45 s."""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
_s = (HERE / '05_mc_tau.py').read_text()
_ch4 = (HERE.parent / 'ch04_statistics' / '07_metropolis_ising.py').read_text()
_ch4 = _ch4[_ch4.index('\n# --- code:metropolis'):]
exec(_ch4[:_ch4.index('def simulate_T')])
_l = _s[_s.index('\n# --- code:mc-tau'):_s.index('\n# --- end of listings')]
exec(_l[:_l.index('rng = np.random.default_rng(seed=2026)')])          # energy_series, tau_int, block_bootstrap
exec(_s[_s.index('def blocking'):_s.index('def row')])


def energy_magnetisation_series(L, T, n_thermalize, n_measure):      # energy_series, also storing M
    lattice = init_lattice(L)
    beta = 1.0 / T
    for _ in range(n_thermalize):
        metropolis_step(lattice, beta)
    E, M = np.empty(n_measure), np.empty(n_measure)
    for k in range(n_measure):
        metropolis_step(lattice, beta)
        E[k] = total_energy(lattice)
        M[k] = lattice.sum()
    return E, M


rng = np.random.default_rng(seed=2026)
L, T = 16, 2.3
E, M = energy_magnetisation_series(L, T, 2000, 20000)
e, m = E / L**2, np.abs(M) / L**2
print(f"<e> = {e.mean():.4f}, tau_int(e) = {tau_int(e)[0]:.1f}")
tau_m, W = tau_int(m)
naive = m.std(ddof=1) / np.sqrt(len(m))
print(f"<|m|> = {m.mean():.3f}, tau_int(|m|) = {tau_m:.1f} (window {W})")
print(f"SE naive {naive:.4f}, with tau {naive * np.sqrt(2 * tau_m):.4f} ({np.sqrt(2 * tau_m):.1f} x)")
for blk in (100, 200, 500, 1000):
    print(f"block bootstrap, blocks of {blk:4d}: {block_bootstrap(m, np.mean, block=blk).std():.4f}")
print("blocking:", blocking(m).round(4))
