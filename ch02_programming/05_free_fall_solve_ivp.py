"""Ch.2  solve_ivp for free fall, checked against the exact solution.

Book references: ssec:solve-ivp, code:solve-ivp, misconception box on default rtol/atol
"""
import numpy as np
from scipy.constants import g
from scipy.integrate import solve_ivp


def free_fall(t, y):
    x, v = y
    return [v, -g]


y0 = [0.0, 20.0]                     # x = 0 m, v = 20 m/s upward
t_span = (0.0, 5.0)
t_eval = np.linspace(*t_span, 200)

sol = solve_ivp(free_fall, t_span, y0, t_eval=t_eval,
                method='RK45', rtol=1e-8, atol=1e-10)
assert sol.success

x_exact = y0[1] * sol.t - 0.5 * g * sol.t**2
print(f"steps accepted: {sol.t.size} output points, {sol.nfev} function evaluations")
print(f"max |x_num - x_exact| = {np.abs(sol.y[0] - x_exact).max():.1e} m")
print("(a constant acceleration is integrated exactly by RK45, so the error is round-off)")

sol_def = solve_ivp(free_fall, t_span, y0, t_eval=t_eval)   # default rtol=1e-3, atol=1e-6
print(f"with default tolerances: max error = {np.abs(sol_def.y[0] - x_exact).max():.1e} m, "
      f"{sol_def.nfev} evaluations")
print("-> for free fall the defaults are fine; 06_oscillator_energy.py shows a case where they are not")
