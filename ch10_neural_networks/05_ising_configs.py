"""Ch.10  16x16 Ising configurations for exercise E10.4 (code:ising-configs, line for line).

Uses init_lattice and metropolis_step from the Chapter 4 listing code:metropolis.
Writes data/ising.npz (spins (3400, 256), T_spin), used by exercises/E10_4_ising_extremes.py.
Numbers quoted in the text: the run takes about one minute; <|m|> ~ 0.58 at T = 2.4.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import ch4_metropolis

os.chdir(HERE)
Path('data').mkdir(exist_ok=True)
ch4_metropolis(globals())      # init_lattice, metropolis_step share this module's rng, as in the book
_t0 = time.time()

# --- code:ising-configs ------------------------------------------------------
# uses init_lattice and metropolis_step from the Chapter 4 listing
def ising_configs(L, T, n_conf, sweeps_between=10, n_thermalize=500):
    lattice = init_lattice(L)
    beta = 1.0 / T
    for _ in range(n_thermalize):
        metropolis_step(lattice, beta)
    out = []
    for _ in range(n_conf):
        for _ in range(sweeps_between):
            metropolis_step(lattice, beta)
        out.append(lattice.copy())
    return np.array(out)

rng = np.random.default_rng(seed=2026)                 # restart the generator
T_ising = np.round(np.arange(1.6, 3.21, 0.1), 1)
spins = np.concatenate([ising_configs(16, T, 200) for T in T_ising])
spins = spins.reshape(len(spins), -1).astype(float)    # (3400, 256)
T_spin = np.repeat(T_ising, 200)
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    print(f"generated {spins.shape} in {time.time() - _t0:.0f} s")
    np.savez('data/ising.npz', spins=spins, T_spin=T_spin)
    print("saved data/ising.npz")
    m = spins.mean(axis=1)
    for T in T_ising:
        k = T_spin == T
        print(f"T = {T:.1f}  <|m|> = {np.abs(m[k]).mean():.2f}  <m> = {m[k].mean():+.2f}")
