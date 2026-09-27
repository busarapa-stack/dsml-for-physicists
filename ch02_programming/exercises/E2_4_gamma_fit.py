"""E2.4  How well does E(t) = A exp(-2 gamma t) recover gamma as damping grows?

Simulate for min(8/gamma, 20) s so the energy falls by a similar fraction in each case.
Book answer: gamma = 0.05 and 0.1 -> error <= ~0.1 %; gamma = 0.5 and 2.0 -> a few %.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oscillator import OMEGA, energy, exp_decay, simulate

print(" gamma   gamma/omega   T_sim   gamma_fit   rel. error")
for gamma in (0.05, 0.1, 0.5, 2.0):
    T = min(8 / gamma, 20.0)
    sol = simulate(gamma, t_end=T)
    E = energy(sol)
    popt, _ = curve_fit(exp_decay, sol.t, E, p0=[E[0], 2 * gamma])
    g_fit = popt[1] / 2
    print(f"{gamma:6.2f}   {gamma / OMEGA:9.4f}   {T:5.1f}   {g_fit:9.5f}   "
          f"{100 * (g_fit - gamma) / gamma:+7.2f} %")
print("-> the exponential is the average trend; it fits well only when gamma << omega (P2b)")
