"""Ch.5  Weighted fit of the Shockley diode equation with physical bounds.

Book references: ssec:shockley, code:shockley-cf (reproduced line for line)
SYNTHETIC data: I0 = 2 pA, n = 1.4, 36 voltages 0.30-0.65 V, 5 % proportional noise, seed 2026.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

os.chdir(Path(__file__).resolve().parent)
Path('outputs').mkdir(exist_ok=True)

# --- code:shockley-cf --------------------------------------------------------
V_T = 0.02585                                   # V, thermal voltage at 300 K

def shockley(V, I0, n):
    return I0 * (np.exp(V / (n * V_T)) - 1)

rng = np.random.default_rng(seed=2026)
V = np.linspace(0.30, 0.65, 36)                 # V
I_obs = shockley(V, 2e-12, 1.4) * (1 + rng.normal(scale=0.05, size=V.size))

p0 = [1e-12, 1.5]
bounds = ([1e-15, 1.0], [1e-9, 2.0])            # I0 in A; n from device physics
popt_d, pcov_d = curve_fit(shockley, V, I_obs, p0=p0, bounds=bounds,
                           sigma=0.05 * I_obs, absolute_sigma=True)

# --- numbers quoted in the text ---------------------------------------------
I0, n = popt_d
e = np.sqrt(np.diag(pcov_d))
rho = pcov_d[0, 1] / (e[0] * e[1])
print(f"I0 = ({I0 * 1e12:.2f} +/- {e[0] * 1e12:.2f}) pA, n = {n:.3f} +/- {e[1]:.3f}")
print(f"correlation(I0, n) = {rho:.3f}")
print(f"V_T = k_B T / q at 300 K = {1.380649e-23 * 300 / 1.602176634e-19 * 1e3:.2f} mV")
print(f"current range: {I_obs.min():.2e} A to {I_obs.max():.2e} A")

# what the book warns about: unweighted fit gives the high-voltage points all the weight
pu, pcu = curve_fit(shockley, V, I_obs, p0=p0, bounds=bounds)
print(f"unweighted fit for comparison: I0 = {pu[0] * 1e12:.2f} pA, n = {pu[1]:.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(V, I_obs, 'o', ms=4, label='data'); ax.semilogy(V, shockley(V, *popt_d), label='weighted fit')
ax.set_xlabel('V (V)'); ax.set_ylabel('I (A)'); ax.legend()
fig.tight_layout(); fig.savefig('outputs/ch05_shockley.png', dpi=150)
print("saved outputs/ch05_shockley.png")
