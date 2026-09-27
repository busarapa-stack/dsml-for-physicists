"""Build the two notebooks of E2.7 (run once by the author; students just open them).

messy_notebook.ipynb   four problems: cells out of order, scattered imports,
                       one name (t) used for two quantities, a plot cell that uses a
                       variable created further down. Its saved outputs are REAL: the
                       cells were executed in a non top-to-bottom order, which is
                       exactly how hidden state is born.
clean_notebook.ipynb   the same analysis in the six-layer structure (def:notebook-6),
                       executed with Restart & Run All.
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

# ---------------------------------------------------------------- messy
messy = nbf.v4.new_notebook()
messy.cells = [
    md("# oscillator + cooling"),
    code("import numpy as np\n"
         "t = np.linspace(0, 20, 400)          # time [s]\n"
         "x = np.exp(-0.1 * t) * np.cos(2 * np.pi * t)"),
    code("import matplotlib.pyplot as plt\n"
         "plt.plot(t, E)\n"
         "plt.xlabel('time (s)'); plt.ylabel('E/m (J/kg)')\n"
         "plt.show()"),
    code("v = np.gradient(x, t)\n"
         "E = 0.5 * v**2 + 0.5 * (2 * np.pi)**2 * x**2"),
    md("cooling data"),
    code("t = 25 + 60 * np.exp(-np.linspace(0, 20, 400) / 6)   # temperature [C]\n"
         "print('max temperature', t.max())"),
    code("from scipy.optimize import curve_fit\n"
         "popt, _ = curve_fit(lambda tt, A, k: A * np.exp(-k * tt), t, E, p0=[E[0], 0.2])\n"
         "print('gamma =', popt[1] / 2)"),
]
# execute in the order a hurried author might have used: 1, 3, 2, 5, 6
order = [1, 3, 2, 5, 6]
client = NotebookClient(messy, timeout=60, kernel_name="python3", allow_errors=True)
with client.setup_kernel():
    for i in order:
        client.execute_cell(messy.cells[i], i)
nbf.write(messy, HERE / "messy_notebook.ipynb")

# ---------------------------------------------------------------- clean
clean = nbf.v4.new_notebook()
clean.cells = [
    md("# Damped oscillator: energy decay rate\n"
       "**Author:** (your name) · **Date:** (date)  \n"
       "**Purpose:** estimate the damping rate $\\gamma$ from the energy of a simulated "
       "damped oscillator and check that it recovers the value we set."),
    md("## 1. Imports (all of them, here only)"),
    code("import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "from scipy.integrate import solve_ivp\n"
         "from scipy.optimize import curve_fit"),
    md("## 2. Settings\nPhysical parameters and numerical tolerances. $\\omega = 2\\pi$ rad/s "
       "(period 1 s), $\\gamma = 0.1\\ \\mathrm{s^{-1}}$."),
    code("OMEGA, GAMMA = 2 * np.pi, 0.1\n"
         "RTOL, ATOL = 1e-9, 1e-11\n"
         "T_END, N_T = 20.0, 4000"),
    md("## 3. Data\nSimulate $\\ddot x = -2\\gamma\\dot x - \\omega^2 x$ and check that "
       "the solver succeeded."),
    code("def ho_rhs(time, y, omega, gamma):\n"
         "    x, v = y\n"
         "    return [v, -2 * gamma * v - omega**2 * x]\n\n"
         "time_s = np.linspace(0, T_END, N_T)          # time [s]  (not 't')\n"
         "sol = solve_ivp(ho_rhs, (0, T_END), [1.0, 0.0], t_eval=time_s,\n"
         "                args=(OMEGA, GAMMA), rtol=RTOL, atol=ATOL)\n"
         "assert sol.success\n"
         "x, v = sol.y\n"
         "print(f'{time_s.size} points, x(0) = {x[0]}, v(0) = {v[0]}')"),
    md("## 4. Analysis\nEnergy per unit mass, then a fit of $E(t) = A e^{-2\\gamma t}$."),
    code("E = 0.5 * v**2 + 0.5 * OMEGA**2 * x**2\n"
         "popt, _ = curve_fit(lambda tt, A, k: A * np.exp(-k * tt), time_s, E, p0=[E[0], 0.2])\n"
         "gamma_fit = popt[1] / 2\n"
         "print(f'gamma set = {GAMMA}, gamma fit = {gamma_fit:.4f}')\n\n"
         "fig, ax = plt.subplots(figsize=(7, 3.5))\n"
         "ax.plot(time_s, E, label='E/m from simulation')\n"
         "ax.plot(time_s, popt[0] * np.exp(-popt[1] * time_s), '--', label='fitted exponential')\n"
         "ax.set_xlabel('time (s)'); ax.set_ylabel('E/m (J/kg)'); ax.legend(); ax.grid(alpha=0.3)\n"
         "plt.show()"),
    md("## 5. Conclusion\nThe fitted $\\gamma$ agrees with the value set to three decimals, so the "
       "simulation and the fit are consistent. **Limitation:** $e^{-2\\gamma t}$ is only the "
       "average trend; it is accurate because $\\gamma \\ll \\omega$ here (see E2.4). "
       "**Next step:** repeat for larger $\\gamma$."),
]
NotebookClient(clean, timeout=60, kernel_name="python3").execute()
nbf.write(clean, HERE / "clean_notebook.ipynb")
print("wrote messy_notebook.ipynb and clean_notebook.ipynb")
