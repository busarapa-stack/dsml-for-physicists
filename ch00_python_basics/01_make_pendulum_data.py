"""Ch.0  Create the SYNTHETIC pendulum data set used throughout Chapter 0.

Book references: sec:py-numpy (the file is read in 06_numpy_matplotlib.py), exercise E0.7
Five string lengths, ten trials each. Each trial is the time for ten complete
swings measured with a hand-held stopwatch, rounded to 0.01 s, as in a first-year
physics laboratory. The "true" value used to generate the data is g = 9.78 m/s^2
(close to the value in Thailand; see the box on scipy.constants.g in Chapter 2).
Reaction-time scatter of the stopwatch: standard deviation 0.15 s per trial.

Choice of SEED (stated openly, see the seed discussion in Chapter 2): with
SEED = 2026 this random draw gives g = 9.63 m/s^2 from the slope of T^2 vs L,
about 3 standard uncertainties from the true 9.78 m/s^2, an unusual draw that
would confuse a first chapter. SEED = 2028 gives a typical draw (g = 9.79).
This is a teaching data set, not a research result.
"""
import math
from pathlib import Path

import numpy as np

SEED = 2028
G_TRUE = 9.78                 # m/s^2
SIGMA_T10 = 0.15              # s, hand timing of 10 swings
LENGTHS = [0.20, 0.40, 0.60, 0.80, 1.00]   # m
N_TRIALS = 10

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

rows = []
for L in LENGTHS:
    T_true = 2 * math.pi * math.sqrt(L / G_TRUE)
    t10 = np.round(10 * T_true + rng.normal(0, SIGMA_T10, N_TRIALS), 2)
    for k, t in enumerate(t10, start=1):
        rows.append((L, k, t))

path = DATA / "pendulum.csv"
with open(path, "w", encoding="utf-8") as f:
    f.write("# SYNTHETIC pendulum data for teaching (g_true = 9.78 m/s^2)\n")
    f.write("# L_m, trial, t10_s  (t10 = time for 10 complete swings)\n")
    for L, k, t in rows:
        f.write(f"{L:.2f},{k},{t:.2f}\n")
print(f"wrote {path.name}: {len(rows)} rows")
for L in LENGTHS:
    vals = [t for (LL, _, t) in rows if LL == L]
    print(f"L = {L:.2f} m  t10 = {[float(v) for v in vals]}")
