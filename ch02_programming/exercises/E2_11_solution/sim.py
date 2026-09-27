"""E2.11  ch02-oscillator: undamped and damped harmonic oscillator with checks.

Run:  python sim.py
Prints the two numerical checks and saves ho_comparison.pdf.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

OMEGA, GAMMA = 2 * np.pi, 0.1          # rad/s, 1/s
Y0 = [1.0, 0.0]                        # m, m/s
T_END, N_T = 20.0, 4000
RTOL, ATOL = 1e-9, 1e-11
HERE = Path(__file__).resolve().parent


def ho_rhs(t, y, omega, gamma):
    x, v = y
    return [v, -2.0 * gamma * v - omega**2 * x]


def simulate(gamma):
    t_eval = np.linspace(0, T_END, N_T)
    sol = solve_ivp(ho_rhs, (0, T_END), Y0, t_eval=t_eval, args=(OMEGA, gamma),
                    method="RK45", rtol=RTOL, atol=ATOL)
    assert sol.success, sol.message
    return sol


def energy(sol):
    return 0.5 * sol.y[1]**2 + 0.5 * OMEGA**2 * sol.y[0]**2


def main():
    sol_un, sol_d = simulate(0.0), simulate(GAMMA)
    E_un, E_d = energy(sol_un), energy(sol_d)
    drift = (E_un.max() - E_un.min()) / E_un[0]
    popt, _ = curve_fit(lambda t, A, k: A * np.exp(-k * t), sol_d.t, E_d, p0=[E_d[0], 0.2])
    print(f"undamped: relative energy drift = {drift:.2e}")
    print(f"damped:   gamma set = {GAMMA:.2f}, gamma fit = {popt[1] / 2:.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for sol, c, lab in ((sol_un, "C0", "undamped"), (sol_d, "C3", f"damped (gamma = {GAMMA} 1/s)")):
        axes[0].plot(sol.t, sol.y[0], color=c, label=lab)
        axes[1].plot(sol.t, sol.y[1], color=c)
        axes[2].plot(sol.t, energy(sol), color=c)
    axes[0].set_ylabel("x (m)"); axes[1].set_ylabel("v (m/s)"); axes[2].set_ylabel("E/m (J/kg)")
    axes[2].set_xlabel("t (s)"); axes[0].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(r"Harmonic oscillator ($\omega = 2\pi$ rad/s)")
    fig.tight_layout()
    fig.savefig(HERE / "ho_comparison.pdf", dpi=300, bbox_inches="tight")
    print("saved ho_comparison.pdf")


if __name__ == "__main__":
    main()
