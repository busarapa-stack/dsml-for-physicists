"""E0.7  g from the slope of T^2 versus L, compared with g from each length."""
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
data = np.loadtxt(HERE / "data" / "pendulum.csv", delimiter=",", comments="#")
L_all, t10_all = data[:, 0], data[:, 2]
L = np.unique(L_all)
T = np.array([np.mean(t10_all[L_all == x]) / 10 for x in L])
slope, intercept = np.polyfit(L, T**2, 1)
print(f"slope fit     : g = {4 * np.pi**2 / slope:.3f} m/s^2 (intercept {intercept:+.4f} s^2)")
g_each = 4 * np.pi**2 * L / T**2
print(f"each length   : {np.round(g_each, 3)}")
print(f"mean of five  : g = {np.mean(g_each):.3f} +/- {np.std(g_each, ddof=1) / np.sqrt(len(g_each)):.3f} m/s^2")
