"""E5.6  Residual analysis: an ideal Shockley fit to a diode that has a series resistance.

SYNTHETIC data: I0 = 2 pA, n = 1.4, R_s = 20 ohm, 46 voltages 0.30-0.75 V (step 0.01 V),
5 % proportional noise, seed 2026.  The implicit equation I = I0 {exp[(V - I R_s)/(n V_T)] - 1}
is solved with the Lambert W function.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import lambertw

os.chdir(Path(__file__).resolve().parent.parent)
Path('data').mkdir(exist_ok=True)
V_T = 0.02585


def shockley(V, I0, n):
    return I0 * (np.exp(V / (n * V_T)) - 1)


def shockley_rs(V, I0, n, Rs):
    """Exact solution of I = I0 (exp((V - I Rs)/(n V_T)) - 1) via Lambert W."""
    a = n * V_T
    arg = I0 * Rs / a * np.exp((V + I0 * Rs) / a)
    return a / Rs * np.real(lambertw(arg)) - I0


rng = np.random.default_rng(seed=2026)
V = np.linspace(0.30, 0.75, 46)
I_true = shockley_rs(V, 2e-12, 1.4, 20.0)
I_obs = I_true * (1 + rng.normal(scale=0.05, size=V.size))
pd.DataFrame({'V': V, 'I_A': I_obs}).to_csv('data/diode_series_R.csv', index=False)
print(f"check: solution satisfies the implicit equation to "
      f"{np.max(np.abs(I_true - shockley(V - I_true * 20.0, 2e-12, 1.4)) / I_true):.1e} (relative)")

# ideal model, as in code:shockley-cf
sig = 0.05 * I_obs
p, pc = curve_fit(shockley, V, I_obs, p0=[1e-12, 1.5], bounds=([1e-15, 1.0], [1e-9, 2.0]),
                  sigma=sig, absolute_sigma=True)
z = (I_obs - shockley(V, *p)) / sig
nu = V.size - 2
print(f"\nideal Shockley: I0 = {p[0] * 1e12:.2f} pA, n = {p[1]:.3f}")
print(f"chi2_red = {np.sum(z**2) / nu:.1f} at nu = {nu}; sqrt(2/nu) = {np.sqrt(2 / nu):.2f}; "
      f"(chi2_red - 1)/sqrt(2/nu) = {(np.sum(z**2) / nu - 1) / np.sqrt(2 / nu):.0f}")
thirds = np.array_split(np.arange(V.size), 3)
print("mean normalised residual in the low / middle / high third of the voltage range: "
      + " / ".join(f"{z[i].mean():+.1f}" for i in thirds))
print(f"most negative normalised residual {z.min():.1f} sigma at V = {V[np.argmin(z)]:.2f} V "
      f"(last point: {z[-1]:.1f} sigma)")

# model with series resistance: three parameters
p3, pc3 = curve_fit(shockley_rs, V, I_obs, p0=[1e-12, 1.5, 10.0],
                    bounds=([1e-15, 1.0, 0.0], [1e-9, 2.0, 1e3]), sigma=sig, absolute_sigma=True)
e3 = np.sqrt(np.diag(pc3))
z3 = (I_obs - shockley_rs(V, *p3)) / sig
print(f"\nwith R_s: I0 = {p3[0] * 1e12:.2f} +/- {e3[0] * 1e12:.2f} pA, n = {p3[1]:.3f} +/- {e3[1]:.3f}, "
      f"R_s = {p3[2]:.1f} +/- {e3[2]:.1f} ohm; chi2_red = {np.sum(z3**2) / (V.size - 3):.2f}")

# only the low-voltage part: is R_s visible there?
low = V <= 0.55
pl, _ = curve_fit(shockley, V[low], I_obs[low], p0=[1e-12, 1.5], bounds=([1e-15, 1.0], [1e-9, 2.0]),
                  sigma=sig[low], absolute_sigma=True)
zl = (I_obs[low] - shockley(V[low], *pl)) / sig[low]
print(f"ideal fit on V <= 0.55 V only: n = {pl[1]:.3f}, chi2_red = {np.sum(zl**2) / (low.sum() - 2):.2f} "
      f"-> R_s is invisible at low current")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(V, z, 'o', ms=4, label='ideal Shockley'); ax.plot(V, z3, 's', ms=3, label='with $R_s$')
ax.axhline(0, color='k', lw=0.8); ax.set_xlabel('V (V)'); ax.set_ylabel(r'$(I_{obs}-\hat I)/\sigma$'); ax.legend()
fig.tight_layout(); fig.savefig('outputs/E5_6_diode_residuals.png', dpi=150)
print("saved outputs/E5_6_diode_residuals.png")
