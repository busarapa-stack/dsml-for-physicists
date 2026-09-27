"""E2.5  Three-panel oscillator figure + Gaussian noise (sigma = 0.03 m, seed 2026),
shaded ±sigma band around the THEORY curve, and a fourth panel: phase space.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oscillator import GAMMA, energy, simulate

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(seed=2026)       # created ONCE, at the top
sigma = 0.03                                 # m

sol_un, sol_d = simulate(0.0), simulate(GAMMA)
x_obs = sol_d.y[0] + rng.normal(0, sigma, sol_d.t.size)

fig, axes = plt.subplots(4, 1, figsize=(7, 10))
ax = axes[0]
ax.plot(sol_d.t, x_obs, ".", ms=1.5, color="C0", label="damped + noise")
ax.plot(sol_d.t, sol_d.y[0], color="C3", lw=1, label="theory")
ax.fill_between(sol_d.t, sol_d.y[0] - sigma, sol_d.y[0] + sigma, color="C3", alpha=0.3,
                label=r"theory $\pm\sigma$")
ax.set_ylabel(r"$x$ (m)"); ax.legend(loc="upper right"); ax.grid(alpha=0.3)
axes[1].plot(sol_un.t, sol_un.y[1], color="C0"); axes[1].plot(sol_d.t, sol_d.y[1], color="C3")
axes[1].set_ylabel(r"$v$ (m/s)"); axes[1].grid(alpha=0.3)
axes[2].plot(sol_un.t, energy(sol_un), color="C0"); axes[2].plot(sol_d.t, energy(sol_d), color="C3")
axes[2].set_ylabel(r"$E/m$ (J/kg)"); axes[2].set_xlabel("t (s)"); axes[2].grid(alpha=0.3)
axes[3].plot(sol_un.y[0], sol_un.y[1], color="C0", label="undamped: closed ellipse")
axes[3].plot(sol_d.y[0], sol_d.y[1], color="C3", lw=0.8, label="damped: inward spiral")
axes[3].set_xlabel(r"$x$ (m)"); axes[3].set_ylabel(r"$v$ (m/s)")
axes[3].legend(loc="upper right"); axes[3].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "ho_with_error_band.pdf", dpi=300, bbox_inches="tight")
inside = np.mean(np.abs(x_obs - sol_d.y[0]) < sigma)
print(f"{100 * inside:.1f} % of the noisy points lie inside the ±sigma band (expected 68.3 %)")
print("saved outputs/ho_with_error_band.pdf")
