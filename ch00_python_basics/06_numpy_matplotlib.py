"""Ch.0  First steps with NumPy arrays and Matplotlib figures.

Book references: sec:py-numpy, code:np-stats, code:loadtxt, code:t2-vs-l, code:euler-plot, exercise E0.7
"""
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# --- code:np-stats -----------------------------------------------------------
t10_s = np.array([17.66, 18.02, 17.76, 17.93, 17.93, 18.28, 17.97, 17.97, 17.90, 18.29])
T_s = t10_s / 10                      # every element divided by 10 at once
N = len(T_s)
T_mean = np.mean(T_s)
s = np.std(T_s, ddof=1)               # sample standard deviation
se = s / np.sqrt(N)                   # standard error of the mean
print(f"T = {T_mean:.4f} +/- {se:.4f} s   (s = {s:.4f} s, N = {N})")
print(f"np.std with ddof=0 gives {np.std(T_s):.4f} s")

# --- code:loadtxt ------------------------------------------------------------
data = np.loadtxt(HERE / "data" / "pendulum.csv", delimiter=",", comments="#")
L_all, t10_all = data[:, 0], data[:, 2]
L = np.unique(L_all)                                   # the five lengths
T_mean_L = np.array([np.mean(t10_all[L_all == x]) / 10 for x in L])
print("L [m]      :", L)
print("T mean [s] :", np.round(T_mean_L, 4))

# --- code:t2-vs-l ------------------------------------------------------------
slope, intercept = np.polyfit(L, T_mean_L**2, 1)       # T^2 = (4 pi^2 / g) L
g_fit = 4 * np.pi**2 / slope
print(f"slope = {slope:.4f} s^2/m  intercept = {intercept:.4f} s^2  g = {g_fit:.3f} m/s^2")

plt.figure(figsize=(5, 3.5))
plt.plot(L, T_mean_L**2, "o", label="measured (mean of 10 trials)")
plt.plot(L, slope * L + intercept, "-", label=f"fit: g = {g_fit:.2f} m/s$^2$")
plt.xlabel("length L [m]")
plt.ylabel("period squared T$^2$ [s$^2$]")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "ch00_t2_vs_L.png", dpi=150)

# --- code:euler-plot ---------------------------------------------------------
g, h0, dt = 9.78, 20.0, 0.1
t_list, y_list = [0.0], [h0]
t, y, v = 0.0, h0, 0.0
while y > 0:
    y, v, t = y + v * dt, v - g * dt, t + dt
    t_list.append(t); y_list.append(y)

t_exact = np.linspace(0, math.sqrt(2 * h0 / g), 200)
plt.figure(figsize=(5, 3.5))
plt.plot(t_exact, h0 - 0.5 * g * t_exact**2, "-", label="exact")
plt.plot(t_list, y_list, "o--", ms=4, label=f"Euler, dt = {dt} s")
plt.xlabel("time t [s]")
plt.ylabel("height y [m]")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "ch00_euler_fall.png", dpi=150)

# --- end of listings ----------------------------------------------------------
(k, b), cov = np.polyfit(L, T_mean_L**2, 1, cov=True)
sk, sb = np.sqrt(np.diag(cov))
print(f"with uncertainties: slope = {k:.3f} +/- {sk:.3f} s^2/m, intercept = {b:.4f} +/- {sb:.4f} s^2, "
      f"g = {4 * np.pi**2 / k:.3f} +/- {4 * np.pi**2 / k * sk / k:.3f} m/s^2")
g_each = 4 * np.pi**2 * L / T_mean_L**2
print("g from each length:", np.round(g_each, 3), " mean =", round(float(np.mean(g_each)), 3))
print("list * 2 :", [1.0, 2.0] * 2, "   array * 2 :", np.array([1.0, 2.0]) * 2)
