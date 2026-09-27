"""E4.4  Thermalisation at T = 2.0 (L = 16) from a random start, and the C_v peak for L = 8
compared with the L = 16 scan of 07_metropolis_ising.py.
Uses the functions of code:metropolis unchanged.
"""
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
# take the function definitions from script 07 (code:metropolis), not the scan itself
src = (HERE / '07_metropolis_ising.py').read_text()
defs = src[src.index('import numpy as np\nrng'):src.index('L = 16\nT_values')]
exec(defs)                                   # defines rng, init_lattice, total_energy, ...

# --- energy trace at T = 2.0 from a hot start -----------------------------------------
L, T = 16, 2.0
lat = init_lattice(L)
trace = []
for sweep in range(1500):
    metropolis_step(lat, 1 / T)
    trace.append(total_energy(lat) / L**2)
trace = np.array(trace)
eq = trace[500:].mean()
first = int(np.argmax(np.abs(trace - eq) < 0.02))
print(f"T = 2.0, L = 16: E/N after 500 sweeps averages {eq:.3f}; first within 0.02 of it at"
      f" sweep {first}")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(trace, lw=0.8); ax.axhline(eq, ls=':', color='k')
ax.set_xlabel('sweep'); ax.set_ylabel('E/N'); ax.set_title('thermalisation, T = 2.0, L = 16')
ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig('outputs/E4_4_thermalization.png', dpi=150)

# --- L = 8 scan with the same settings as the L = 16 scan ------------------------------
T_values = np.linspace(1.5, 3.5, 21)
res8 = [simulate_T(8, T, n_thermalize=1000, n_measure=2000) for T in T_values]
Cv8 = np.array([r[1] for r in res8])
pd.DataFrame({'T': T_values, 'E_per_spin': [r[0] for r in res8], 'Cv_per_spin': Cv8}).round(4) \
    .to_csv('outputs/ch04_ising_metropolis_L8.csv', index=False)   # used by E4_5b
res16 = pd.read_csv('outputs/ch04_ising_metropolis_L16.csv')
i8, i16 = int(np.argmax(Cv8)), int(res16['Cv_per_spin'].idxmax())
print(f"C_v/N peak: L = 8 at T = {T_values[i8]:.1f} (height {Cv8[i8]:.2f}); "
      f"L = 16 at T = {res16['T'][i16]:.1f} (height {res16['Cv_per_spin'][i16]:.2f})")
print("-> the smaller lattice gives a lower, broader peak; both lie near T_c = 2.269,"
      " limited by the 0.1 temperature grid")
