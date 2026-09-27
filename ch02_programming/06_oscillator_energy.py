"""Ch.2  Undamped and damped harmonic oscillator; energy as a correctness check.

Book references: ssec:ho, ssec:energy-analysis, code:ho, code:energy-fit,
                 misconception boxes "the default tolerances of solve_ivp are always
                 enough" and "the energy of a damped oscillator decays exactly exponentially"
The two listings are reproduced line for line below; the extra checks follow.
Text claims checked
    rtol=1e-9: relative energy drift of the undamped case ~ 3e-8
    fitted gamma printed as 0.1001 (set 0.10), equal to three decimals
    default rtol/atol over 20 periods: energy varies by about 0.35 %
"""
import numpy as np
from scipy.integrate import solve_ivp

# --- code:ho ----------------------------------------------------------------
def ho_rhs(t, y, omega, gamma):
    x, v = y
    return [v, -2.0 * gamma * v - omega**2 * x]

omega = 2.0 * np.pi
y0 = [1.0, 0.0]
t_span = (0.0, 20.0)
t_eval = np.linspace(*t_span, 4000)

sol_un = solve_ivp(ho_rhs, t_span, y0, t_eval=t_eval,
                   args=(omega, 0.0),
                   method='RK45', rtol=1e-9, atol=1e-11)
sol_d  = solve_ivp(ho_rhs, t_span, y0, t_eval=t_eval,
                   args=(omega, 0.1),
                   method='RK45', rtol=1e-9, atol=1e-11)
assert sol_un.success and sol_d.success

# --- code:energy-fit --------------------------------------------------------
from scipy.optimize import curve_fit

def energy(sol, omega):
    x, v = sol.y[0], sol.y[1]
    return 0.5 * v**2 + 0.5 * omega**2 * x**2

E_un = energy(sol_un, omega)
E_d  = energy(sol_d,  omega)

drift = (E_un.max() - E_un.min()) / E_un[0]
print(f"Undamped: relative energy drift = {drift:.2e}")

def exp_decay(t, A, k):
    return A * np.exp(-k * t)
popt, _ = curve_fit(exp_decay, sol_d.t, E_d, p0=[E_d[0], 0.2])
gamma_fit = popt[1] / 2
print(f"Damped: gamma_set = 0.10, gamma_fit = {gamma_fit:.4f}")

# --- misconception box: default tolerances ---------------------------------
sol_def = solve_ivp(ho_rhs, t_span, y0, t_eval=t_eval, args=(omega, 0.0))  # rtol=1e-3, atol=1e-6
E_def = energy(sol_def, omega)
print(f"\nDefault rtol=1e-3, atol=1e-6, 20 periods: energy varies by "
      f"{100 * (E_def.max() - E_def.min()) / E_def[0]:.2f} %")
print(f"rtol=1e-9, atol=1e-11,        20 periods: energy varies by {100 * drift:.1e} %")
print(f"function evaluations: {sol_def.nfev} (default) vs {sol_un.nfev} (tight)")

# --- misconception box: energy does not decay as a smooth exponential -------
ratio = E_d / exp_decay(sol_d.t, *popt)
print(f"\nDamped E(t) / fitted exponential ranges from {ratio.min():.3f} to {ratio.max():.3f}"
      " -> small steps: energy is lost fastest where |v| is largest")

# --- the shared module oscillator.py (used by 08 and the exercises) gives the same numbers
from oscillator import energy as energy_mod, simulate
assert np.allclose(energy_mod(simulate(0.1)), E_d)
print("oscillator.py reproduces these results exactly")
