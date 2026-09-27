"""Ch.2  Matplotlib figures: object-oriented basics, error bars, shaded bands,
multi-panel figure, and saving for publication.

Book references: sec:matplotlib, code:mpl-oo, code:errorbar, code:fillbetween,
                 code:multipanel, tab:fig-format
The book shows only the plotting lines of code:errorbar and code:fillbetween;
the data they use (t_obs, x_obs, sigma, t_dense, ...) are created here.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oscillator import GAMMA, OMEGA, energy, simulate

SEED = 2026
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)
omega = OMEGA

# --- code:mpl-oo ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
t = np.linspace(0, 2*np.pi, 200)
ax.plot(t, np.sin(t), label=r'$\sin(t)$')
ax.plot(t, np.cos(t), label=r'$\cos(t)$', linestyle='--')
ax.set_xlabel('t (rad)')
ax.set_ylabel('amplitude')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "ch02_mpl_oo.png", dpi=150); plt.close(fig)

# --- code:errorbar : few points, one error bar each -------------------------
sigma = 0.08                                       # m
t_obs = np.linspace(0, 2, 21)                      # 21 measurements
x_obs = np.cos(omega * t_obs) + rng.normal(0, sigma, t_obs.size)
t_fine = np.linspace(0, 2, 500)

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(t_obs, x_obs, yerr=sigma,
            fmt='o', capsize=3, label='measured', color='C0')
ax.plot(t_fine, np.cos(omega*t_fine),
        label=r'theory: $\cos(\omega t)$', color='C3')
ax.legend(); ax.grid(alpha=0.3)
ax.set_xlabel('t (s)'); ax.set_ylabel('x (m)')
ax.set_title(r'error bars = $\pm 1\sigma$ of each measurement ($\sigma$ = 0.08 m)')
fig.tight_layout(); fig.savefig(OUT / "ch02_errorbar.png", dpi=150); plt.close(fig)
inside = np.abs(x_obs - np.cos(omega * t_obs)) < sigma
n_exp = 0.683 * t_obs.size
n_sd = np.sqrt(t_obs.size * 0.683 * 0.317)          # binomial spread
print(f"errorbar: {t_obs.size} points, {inside.sum()} within 1 sigma of theory "
      f"(expected {n_exp:.0f} +- {n_sd:.0f}; with only 21 points this count fluctuates a lot)")

# --- code:fillbetween : dense data, shaded band -----------------------------
t_dense = np.linspace(0, 2, 2000)
x_dense_true = np.cos(omega * t_dense)
sigma_band = 0.05 + 0.05 * t_dense / 2             # noise that grows with time
x_dense_obs = x_dense_true + rng.normal(0, sigma_band)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_dense, x_dense_obs, color='C0', lw=0.6, label='data')
ax.plot(t_dense, x_dense_true, color='C3', lw=1.5, label='theory')
ax.fill_between(t_dense,
                x_dense_true - sigma_band,
                x_dense_true + sigma_band,
                alpha=0.3, color='C3', label=r'$\pm\sigma(t)$')
ax.legend(); ax.grid(alpha=0.3)
ax.set_xlabel('t (s)'); ax.set_ylabel('x (m)')
fig.tight_layout(); fig.savefig(OUT / "ch02_fillbetween.png", dpi=150); plt.close(fig)
frac = np.mean(np.abs(x_dense_obs - x_dense_true) < sigma_band)
print(f"fill_between: {t_dense.size} points, {100 * frac:.1f} % inside the ±sigma band")

# --- code:multipanel --------------------------------------------------------
sol_un, sol_d = simulate(0.0), simulate(GAMMA)
fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

axes[0].plot(sol_un.t, sol_un.y[0], label='undamped', color='C0')
axes[0].plot(sol_d.t,  sol_d.y[0],  label='damped (gamma = 0.1 1/s)', color='C3')
axes[0].set_ylabel(r'$x(t)$ (m)'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(sol_un.t, sol_un.y[1], color='C0')
axes[1].plot(sol_d.t,  sol_d.y[1],  color='C3')
axes[1].set_ylabel(r'$v(t)$ (m/s)'); axes[1].grid(alpha=0.3)

axes[2].plot(sol_un.t, energy(sol_un, omega), color='C0')
axes[2].plot(sol_d.t,  energy(sol_d,  omega), color='C3')
axes[2].set_ylabel(r'$E(t)/m$ (J/kg)')
axes[2].set_xlabel(r'$t$ (s)'); axes[2].grid(alpha=0.3)

fig.suptitle(r'Harmonic oscillator ($\omega = 2\pi$ rad/s)')
plt.tight_layout()

# --- saving for publication (tab:fig-format) --------------------------------
fig.savefig(OUT / 'ho_comparison.pdf', dpi=300, bbox_inches='tight')   # book: fig.savefig('ho_comparison.pdf', ...)
fig.savefig(OUT / 'ho_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
for f in ("ch02_mpl_oo.png", "ch02_errorbar.png", "ch02_fillbetween.png",
          "ho_comparison.pdf", "ho_comparison.png"):
    print(f"saved outputs/{f:24s} {(OUT / f).stat().st_size / 1024:7.1f} kB")
