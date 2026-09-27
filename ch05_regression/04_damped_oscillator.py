"""Ch.5  Chi-squared fit of a damped oscillator with physical bounds.

Book references: ssec:damped-fit, ssec:fitted-PI, ssec:chi2-red, code:damped-cf (line for line)
SYNTHETIC data: A = 1.0 m, gamma = 0.3 1/s, omega = 6.0 rad/s, phi = 0.5 rad, 200 points in 10 s,
noise 0.05 m, seed 2026.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

os.chdir(Path(__file__).resolve().parent)
Path('data').mkdir(exist_ok=True); Path('outputs').mkdir(exist_ok=True)

# --- code:damped-cf ----------------------------------------------------------
from scipy.optimize import curve_fit

def damped_oscillator(t, A, gamma, omega, phi):
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi)

rng = np.random.default_rng(seed=2026)
t = np.linspace(0, 10, 200)                               # s
sigma_x = 0.05                                            # m
x_obs = damped_oscillator(t, 1.0, 0.3, 6.0, 0.5) \
        + rng.normal(scale=sigma_x, size=t.size)

p0 = [1.0, 0.2, 6.0, 0.0]                                 # initial guess
bounds = ([0, 0, 0, -np.pi], [10, 5, 20, np.pi])          # physical limits
popt, pcov = curve_fit(damped_oscillator, t, x_obs,
                       p0=p0, bounds=bounds,
                       sigma=sigma_x * np.ones(t.size),
                       absolute_sigma=True)

A, gamma, omega, phi = popt
errs = np.sqrt(np.diag(pcov))
print(f"gamma = {gamma:.4f} +/- {errs[1]:.4f} 1/s")
print(f"omega = {omega:.4f} +/- {errs[2]:.4f} rad/s")

# --- numbers quoted in the text ---------------------------------------------
pd.DataFrame({'t_s': t, 'x_m': x_obs}).to_csv('data/damped_oscillator.csv', index=False)
print(f"\nA = {A:.4f} +/- {errs[0]:.4f} m, phi = {phi:.4f} +/- {errs[3]:.4f} rad")
print(f"omega is {(omega - 6.0) / errs[2]:+.1f} sigma from 6.0, gamma is {(gamma - 0.3) / errs[1]:+.1f} sigma from 0.3")

resid = x_obs - damped_oscillator(t, *popt)
chi2_min = np.sum((resid / sigma_x)**2)
nu = t.size - len(popt)
print(f"chi2 = {chi2_min:.1f}, nu = {nu}, chi2_red = {chi2_min / nu:.2f}, "
      f"sqrt(2/nu) = {np.sqrt(2 / nu):.2f}, p = {stats.chi2.sf(chi2_min, nu):.3f}")

omega0 = np.sqrt(omega**2 + gamma**2)
print(f"Q = omega0/(2 gamma) = {omega0 / (2 * gamma):.2f} (true {np.sqrt(36 + 0.09) / 0.6:.2f}); "
      f"1/gamma = {1 / gamma:.2f} s; oscillations in 1/gamma: {omega / gamma / (2 * np.pi):.1f}; "
      f"gamma/omega = {gamma / omega:.3f}; (omega0 - omega)/omega = {100 * (omega0 / omega - 1):.2f} %")

# multi-start: does the fit converge to the same minimum from other starting omegas?
same = []
for w0 in np.arange(1, 19):
    try:
        p, _ = curve_fit(damped_oscillator, t, x_obs, p0=[1.0, 0.2, w0, 0.0], bounds=bounds,
                         sigma=sigma_x * np.ones(t.size), absolute_sigma=True)
        same.append(abs(p[2] - omega) < 1e-3)
    except RuntimeError:
        same.append(False)
ok = np.arange(1, 19)[np.array(same)]
print(f"starting omega 1..18 rad/s: converged to omega = {omega:.3f} from {len(ok)} of 18 starts "
      f"({'all' if len(ok) == 18 else list(ok)})")

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
a1.plot(t, x_obs, '.', ms=3, label='data'); a1.plot(t, damped_oscillator(t, *popt), label='fit')
a1.set_ylabel('x (m)'); a1.legend()
a2.plot(t, resid / sigma_x, '.', ms=3); a2.axhline(0, color='k', lw=0.8)
a2.set_xlabel('t (s)'); a2.set_ylabel(r'residual / $\sigma$')
fig.tight_layout(); fig.savefig('outputs/ch05_damped_fit.png', dpi=150)
print("saved outputs/ch05_damped_fit.png")
