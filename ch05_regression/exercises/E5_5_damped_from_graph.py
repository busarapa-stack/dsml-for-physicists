"""E5.5  Starting values read from the graph, the fit, Q, and the residuals of the damped oscillator."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

os.chdir(Path(__file__).resolve().parent.parent)
if not Path('data/damped_oscillator.csv').exists():
    os.system('python 04_damped_oscillator.py > /dev/null')
d = pd.read_csv('data/damped_oscillator.csv')
t, x = d['t_s'].to_numpy(), d['x_m'].to_numpy()


def damped_oscillator(t, A, gamma, omega, phi):
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi)


# 1. starting values from the graph
pk, _ = find_peaks(x, distance=15)                # maxima at least ~0.75 s apart
cycles = len(pk) - 1
w_guess = 2 * np.pi * cycles / (t[pk[-1]] - t[pk[0]])
A_guess = x[pk[0]]
g_guess = np.log(x[pk[0]] / x[pk[4]]) / (t[pk[4]] - t[pk[0]])
print(f"{len(pk)} maxima; omega from their spacing = {w_guess:.2f} rad/s; "
      f"A from the first maximum = {A_guess:.2f} m; gamma from maxima 1 and 5 = {g_guess:.2f} 1/s")

# 2. fit
popt, pcov = curve_fit(damped_oscillator, t, x, p0=[A_guess, g_guess, w_guess, 0.0],
                       bounds=([0, 0, 0, -np.pi], [10, 5, 20, np.pi]),
                       sigma=0.05 * np.ones(t.size), absolute_sigma=True)
A, g, w, phi = popt
e = np.sqrt(np.diag(pcov))
print(f"gamma = ({g:.3f} +/- {e[1]:.3f}) 1/s, omega = ({w:.3f} +/- {e[2]:.3f}) rad/s")
print(f"relative uncertainty: gamma {100 * e[1] / g:.1f} %, omega {100 * e[2] / w:.2f} %")

# 3. Q with full propagation
J = np.array([0.0, -w / (2 * g**2), 1 / (2 * g), 0.0])
Q, sQ = w / (2 * g), np.sqrt(J @ pcov @ J)
Q_true = np.sqrt(6.0**2 + 0.3**2) / (2 * 0.3)
print(f"Q = {Q:.1f} +/- {sQ:.1f}; true Q = omega0/(2 gamma) = {Q_true:.2f}; "
      f"difference {(Q - Q_true) / sQ:.1f} sigma")

# 4. residuals
r = x - damped_oscillator(t, *popt)
print(f"residuals: mean {r.mean():+.4f} m, std {r.std(ddof=4):.4f} m (noise 0.05 m); "
      f"lag-1 autocorrelation {np.corrcoef(r[:-1], r[1:])[0, 1]:+.2f}")
