"""Ch.4  Metropolis Monte Carlo for the 2D Ising model (L = 16), temperature scan.

Book references: ssec:metropolis, code:metropolis, tab:ising-T-scan, exercise E4.4
The listing is reproduced line for line (it takes about 2 minutes in pure Python).
Results are compared with Onsager's exact infinite-lattice energy and saved to
outputs/ch04_ising_metropolis_L16.csv for exercise E4.4 and script 08.
"""
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import ellipk

os.chdir(Path(__file__).resolve().parent)
t0 = time.perf_counter()

# --- code:metropolis --------------------------------------------------------
import numpy as np
rng = np.random.default_rng(seed=2026)

def init_lattice(L, hot_start=True):
    if hot_start:
        return rng.choice([-1, 1], size=(L, L))
    return np.ones((L, L), dtype=int)

def total_energy(lattice):
    # each bond counted once: one neighbour along each axis
    return -np.sum(lattice * (np.roll(lattice, 1, 0) + np.roll(lattice, 1, 1)))

def metropolis_step(lattice, beta):
    """One sweep = L*L attempted single-spin flips."""
    L = lattice.shape[0]
    for _ in range(L * L):
        i, j = rng.integers(0, L, size=2)
        s = lattice[i, j]
        nb = (lattice[(i+1) % L, j] + lattice[(i-1) % L, j]
            + lattice[i, (j+1) % L] + lattice[i, (j-1) % L])
        dE = 2 * s * nb
        if dE <= 0 or rng.random() < np.exp(-beta * dE):
            lattice[i, j] = -s
    return lattice

def simulate_T(L, T, n_thermalize, n_measure):
    lattice = init_lattice(L)
    beta = 1.0 / T
    for _ in range(n_thermalize):          # discard: not yet in equilibrium
        metropolis_step(lattice, beta)
    energies = []
    for _ in range(n_measure):
        metropolis_step(lattice, beta)
        energies.append(total_energy(lattice))
    energies = np.asarray(energies)
    E_mean = energies.mean() / (L * L)
    Cv = energies.var() / (T**2 * L * L)   # fluctuation formula
    return E_mean, Cv

L = 16
T_values = np.linspace(1.5, 3.5, 21)
E_means, Cvs = [], []
for T in T_values:
    Em, Cv = simulate_T(L, T, n_thermalize=1000, n_measure=2000)
    E_means.append(Em); Cvs.append(Cv)

# --- results --------------------------------------------------------------------
elapsed = time.perf_counter() - t0


def onsager_energy(T):
    """Exact energy per spin of the infinite 2D Ising model (J = k_B = 1)."""
    b = 1.0 / T
    k = 2 * np.sinh(2 * b) / np.cosh(2 * b)**2
    return -(1 / np.tanh(2 * b)) * (1 + 2 / np.pi * (2 * np.tanh(2 * b)**2 - 1) * ellipk(k**2))


Tc = 2 / np.log(1 + np.sqrt(2))
res = pd.DataFrame({'T': T_values, 'E_per_spin': E_means, 'Cv_per_spin': Cvs,
                    'E_onsager': [onsager_energy(T) for T in T_values]})
res.round(4).to_csv('outputs/ch04_ising_metropolis_L16.csv', index=False)
print(res.round(3).to_string(index=False))
print(f"\nrun time: {elapsed:.0f} s for 21 temperatures")
print(f"T_c (Onsager) = {Tc:.4f}; exact E/N just above T_c = {onsager_energy(Tc * (1 + 1e-9)):.4f}, -sqrt(2) = {-np.sqrt(2):.4f}")
i = int(np.argmax(Cvs))
print(f"C_v peak of this L = 16 run at T = {T_values[i]:.2f} (grid step 0.1), C_v/N = {Cvs[i]:.2f}")
for T in (1.5, 2.0, 3.5):
    j = int(np.argmin(abs(T_values - T)))
    print(f"E/N at T = {T}: {E_means[j]:.3f} (Onsager, infinite lattice: {onsager_energy(T):.3f})")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
Tf = np.linspace(1.5, 3.5, 200)
ax[0].plot(T_values, E_means, 'o', label='Metropolis, L = 16')
ax[0].plot(Tf, [onsager_energy(T) for T in Tf], '-', label='Onsager, L = infinity')
ax[0].set_xlabel('T'); ax[0].set_ylabel('<E>/N'); ax[0].legend()
ax[1].plot(T_values, Cvs, 'o-'); ax[1].axvline(Tc, ls=':', color='gray')
ax[1].set_xlabel('T'); ax[1].set_ylabel('C_v/N')
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('outputs/ch04_ising_metropolis.png', dpi=150)
print("saved outputs/ch04_ising_metropolis.png and outputs/ch04_ising_metropolis_L16.csv")
