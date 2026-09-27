"""Ch.5  Error propagation with the full covariance matrix: quality factor Q and amplitude at 5 s.

Book references: ssec:err-prop, ex:Q-prop, code:q-prop (reproduced line for line)
Runs the fit of code:damped-cf first (taken from 04_damped_oscillator.py) to get popt, pcov, errs.
"""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
src = (HERE / '04_damped_oscillator.py').read_text()
exec(src[src.index('from scipy.optimize import curve_fit'):src.index('# --- numbers quoted in the text')])

# --- code:q-prop -------------------------------------------------------------
Q = omega / (2 * gamma)
dQ_dom = 1 / (2 * gamma)
dQ_dga = -omega / (2 * gamma**2)

# naive: ignore the off-diagonal term
s_naive = np.sqrt(dQ_dom**2 * pcov[2, 2] + dQ_dga**2 * pcov[1, 1])

# full: include the covariance between gamma (index 1) and omega (index 2)
s_full = np.sqrt(dQ_dom**2 * pcov[2, 2] + dQ_dga**2 * pcov[1, 1]
                 + 2 * dQ_dom * dQ_dga * pcov[1, 2])

# matrix form, J ordered as (A, gamma, omega, phi)
J = np.array([0.0, dQ_dga, dQ_dom, 0.0])
s_matrix = np.sqrt(J @ pcov @ J)       # identical to s_full

rho = pcov / np.outer(errs, errs)      # correlation matrix

# --- numbers quoted in the text ---------------------------------------------
print(f"\nQ = {Q:.2f}; sigma_Q naive = {s_naive:.3f}, full = {s_full:.3f}, matrix = {s_matrix:.3f}")
print(f"Q (true, omega/(2 gamma) with 6.0 and 0.3) = {6.0 / 0.6:.2f}; distance {(Q - 10.0) / s_full:.1f} sigma")
np.set_printoptions(precision=2, suppress=True)
print("correlation matrix (A, gamma, omega, phi):")
print(rho)
print(f"rho(gamma, omega) = {rho[1, 2]:+.2f}, rho(A, gamma) = {rho[0, 1]:+.2f}, rho(omega, phi) = {rho[2, 3]:+.2f}")

# amplitude at t = 5 s: a = A exp(-5 gamma)
t5 = 5.0
a5 = A * np.exp(-gamma * t5)
Ja = np.array([np.exp(-gamma * t5), -t5 * A * np.exp(-gamma * t5), 0.0, 0.0])
s_a_naive = np.sqrt(np.sum(Ja**2 * np.diag(pcov)))
s_a_full = np.sqrt(Ja @ pcov @ Ja)
print(f"a(5 s) = {a5:.3f} m; sigma naive = {s_a_naive:.4f} m, full = {s_a_full:.4f} m "
      f"(naive / full = {s_a_naive / s_a_full:.2f})")

# check the linear propagation against a Monte Carlo draw from N(popt, pcov)
draws = np.random.default_rng(1).multivariate_normal(popt, pcov, size=200_000)
print(f"Monte Carlo from N(popt, pcov): sigma_Q = {np.std(draws[:, 2] / (2 * draws[:, 1])):.3f}, "
      f"sigma_a(5 s) = {np.std(draws[:, 0] * np.exp(-5 * draws[:, 1])):.4f} m")
