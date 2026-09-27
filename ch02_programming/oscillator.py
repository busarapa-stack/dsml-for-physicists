"""Shared helpers for Ch.2: the harmonic oscillator of Section 2.2 (code:ho, code:energy-fit).

Imported by 06_oscillator_energy.py, 08_figures.py and several exercises so that
every script uses exactly the same model.
"""
import numpy as np
from scipy.integrate import solve_ivp

OMEGA = 2.0 * np.pi          # rad/s, period 1 s
GAMMA = 0.1                  # 1/s, damped case
Y0 = [1.0, 0.0]              # x = 1 m, v = 0


def ho_rhs(t, y, omega, gamma):
    """dx/dt = v,  dv/dt = -2 gamma v - omega^2 x"""
    x, v = y
    return [v, -2.0 * gamma * v - omega**2 * x]


def simulate(gamma, t_end=20.0, n=4000, omega=OMEGA, rtol=1e-9, atol=1e-11, **kw):
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(ho_rhs, (0.0, t_end), Y0, t_eval=t_eval, args=(omega, gamma),
                    method="RK45", rtol=rtol, atol=atol, **kw)
    assert sol.success, sol.message
    return sol


def energy(sol, omega=OMEGA):
    """Energy per unit mass, E/m = v^2/2 + omega^2 x^2/2  [J/kg]"""
    x, v = sol.y[0], sol.y[1]
    return 0.5 * v**2 + 0.5 * omega**2 * x**2


def exp_decay(t, A, k):
    return A * np.exp(-k * t)
