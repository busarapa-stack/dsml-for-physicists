"""E2.3  Energy conservation versus the tolerance rtol, 100 periods, undamped oscillator.

Book answer: the energy error grows with time and shrinks as rtol shrinks;
choose rtol from the plot, not by guessing.
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oscillator import energy, simulate

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 4))
print(" rtol    atol   max|dE/E0| (100 T)  at 10 T    first > 1e-6   evals   time")
for rtol in (1e-3, 1e-6, 1e-9):
    t0 = time.perf_counter()
    sol = simulate(0.0, t_end=100.0, n=20000, rtol=rtol, atol=rtol * 1e-2)
    dt = time.perf_counter() - t0
    E = energy(sol)
    err = np.abs(E - E[0]) / E[0]
    over = sol.t[err > 1e-6]
    first = f"t = {over[0]:6.2f} s" if over.size else "never     "
    print(f"{rtol:5.0e}  {rtol * 1e-2:5.0e}   {err.max():10.2e}        "
          f"{err[sol.t <= 10].max():9.2e}  {first}  {sol.nfev:6d}  {dt:5.2f} s")
    ax.semilogy(sol.t, np.maximum(err, 1e-16), label=f"rtol = {rtol:.0e}")
ax.axhline(1e-6, color="k", ls=":", label="target 1e-6")
ax.set_xlabel("t (s)  [1 period = 1 s]"); ax.set_ylabel(r"$|E - E_0| / E_0$")
ax.legend(); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "E2_3_energy_error.png", dpi=150)
print("-> the error grows about linearly with time; for |dE/E0| <= 1e-6 over 100 periods"
      " only rtol = 1e-9 works (rtol = 1e-6 already fails in the first period)")
print("saved outputs/E2_3_energy_error.png")
