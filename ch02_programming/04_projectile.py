"""Ch.2  Vectorized projectile trajectories for several angles, with sanity checks.

Book references: ssec:projectile, code:projectile
Checks from the text
    v0 = 20 m/s, theta = 45 deg  ->  R = v0^2 / g = 400 / 9.80665 = 40.79 m
    15/75 deg and 30/60 deg land at the same distance (sin 2θ = sin(180° − 2θ))
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


def projectile(v0, theta_deg, n=1000):
    theta = np.deg2rad(theta_deg)
    T = 2.0 * v0 * np.sin(theta) / g
    t = np.linspace(0.0, T, n)
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2
    return t, x, y


v0 = 20.0
fig, ax = plt.subplots(figsize=(7, 4))
ranges = {}
for theta in (15, 30, 45, 60, 75):
    _, x, y = projectile(v0, theta)
    ranges[theta] = x[-1]
    ax.plot(x, y, label=f"theta = {theta} deg")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "ch02_projectile.png", dpi=150)

R_formula = v0**2 / g
print(f"R(45 deg) from code  = {ranges[45]:.4f} m")
print(f"R = v0^2/g = 400/{g} = {R_formula:.4f} m  -> match: {np.isclose(ranges[45], R_formula)}")
for a, b in ((15, 75), (30, 60)):
    print(f"R({a}) = {ranges[a]:.4f} m, R({b}) = {ranges[b]:.4f} m  -> equal: "
          f"{np.isclose(ranges[a], ranges[b])}")
print("saved outputs/ch02_projectile.png")
