"""E9.4  Posterior of the poor data set (60 points in 10 s, sigma 0.4 m, seed 2026) with the sampler of code:mcmc-hand,
100,000 steps (burn-in 10,000).  About 10 s."""
import os
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
_s = (HERE / '05_mc_tau.py').read_text()
exec(_s[_s.index('def tau_int'):_s.index('def block_bootstrap')])


def damped_oscillator(t, A, gamma, omega, phi):          # as in code:damped-cf
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi)


rng = np.random.default_rng(seed=2026)
t = np.linspace(0, 10, 60)
sigma_x = 0.4
x_obs = damped_oscillator(t, 1.0, 0.3, 6.0, 0.5) + rng.normal(scale=sigma_x, size=t.size)
bounds = ([0, 0, 0, -np.pi], [10, 5, 20, np.pi])
popt, pcov = curve_fit(damped_oscillator, t, x_obs, p0=[1.0, 0.2, 6.0, 0.0], bounds=bounds,
                       sigma=sigma_x * np.ones(t.size), absolute_sigma=True)
print(f"curve_fit: gamma = {popt[1]:.3f} +/- {np.sqrt(pcov[1, 1]):.3f}, Q = {popt[2] / (2 * popt[1]):.1f}")

_l = (HERE / '07_mcmc_hand.py').read_text()
_l = _l[_l.index('\n# --- code:mcmc-hand'):_l.index('\n# --- end of listings')]
exec(_l.replace('range(50000)', 'range(100000)').replace('n_acc / 50000', 'n_acc / 100000').replace('[5000:]', '[10000:]'))
Q = chain[:, 2] / (2 * chain[:, 1])
q = np.percentile(Q, [2.5, 16, 50, 84, 97.5])
print(f"Q: median {q[2]:.1f}, 95 % [{q[0]:.1f}, {q[4]:.1f}], report {q[2]:.1f} +{q[3] - q[2]:.1f} -{q[2] - q[1]:.1f}")
print(f"independent samples of gamma: {len(chain) / (2 * tau_int(chain[:, 1])[0]):.0f}")
