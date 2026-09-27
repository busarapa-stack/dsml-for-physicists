"""Ch.4  Reporting a measurement: rounding rules, stat/syst split, error propagation.

Book references: ssec:reporting, tab:reporting-examples, ssec:uq-PI, exercise E4.6
Rule used in the book (Taylor): round the uncertainty to one significant figure, or two
if its first digit is 1; then round the value to the same decimal place.
"""
import math

import numpy as np


def report(value, unc, unit=""):
    """Format value +/- uncertainty following the rule in ssec:reporting."""
    exp10 = math.floor(math.log10(abs(unc)))
    first = int(unc / 10**exp10)
    digits = 2 if first == 1 else 1
    dec = -(exp10 - digits + 1)                         # decimal places to keep
    u = round(unc, dec); v = round(value, dec)
    if abs(v) >= 1e4 or abs(v) < 1e-3:                  # scientific notation
        e = math.floor(math.log10(abs(v)))
        s = 10**e
        d = max(dec + e, 0)
        return f"({v / s:.{d}f} ± {u / s:.{d}f}) × 10^{e} {unit}".strip()
    d = max(dec, 0)
    return f"({v:.{d}f} ± {u:.{d}f}) {unit}".strip()


print("tab:reporting-examples, recomputed from the 'wrong' versions:")
for v, u, unit in [(656.28135, 0.00521, "nm"), (9.81254, 0.02, "m/s^2"),
                   (342156.7, 15234.2, "m/s")]:
    print(f"  {v} ± {u}  ->  {report(v, u, unit)}")
print(f"rounding 0.14 to 0.1 changes it by {100 * 0.04 / 0.14:.0f} % (text: almost 30 %)")

# --- Co-60 half-life (text and E4.6) -------------------------------------------------
T, stat, syst = 5.2713, 0.0042, 0.005
print(f"\nT1/2(Co-60) = {T:.3f} ± {round(stat, 3)} (stat) ± {syst} (syst) years")
ref_d, ref_ud = 1925.28, 0.14
ref, uref = ref_d / 365.25, ref_ud / 365.25
print(f"reference: {ref_d} ± {ref_ud} d = {ref:.4f} ± {uref:.4f} y (1 y = 365.25 d)")
for days in (365.2422, 365.0):
    shift = ref_d / days - ref
    print(f"  with 1 y = {days} d: {ref_d / days:.4f} y, shift {shift:+.4f} y "
          f"({'larger' if abs(shift) > uref else 'smaller'} than the reference uncertainty {uref:.4f} y)")
tot = math.hypot(stat, syst)
print(f"95 % CI (statistical only): {T} ± {1.96 * stat:.4f} -> [{T - 1.96 * stat:.3f}, {T + 1.96 * stat:.3f}] y")
print(f"difference measured - reference = {T - ref:+.4f} y = {(T - ref) / stat:+.2f} sigma_stat")
print(f"combined uncertainty sqrt(stat^2 + syst^2) = {tot:.4f} y; halving stat (4x more counting)"
      f" gives {math.hypot(stat / 2, syst):.4f} y: only {100 * (1 - math.hypot(stat / 2, syst) / tot):.0f} % better")

# --- error propagation: pendulum g = 4 pi^2 l / T^2 -------------------------------------
l, sl = 1.000, 0.001            # m
Tp, sT = 2.007, 0.02 * 2.007 / 2  # s, 1 % uncertainty
g = 4 * np.pi**2 * l / Tp**2
sg = g * np.hypot(sl / l, 2 * sT / Tp)
print(f"\npendulum: T known to 1 %, l to 0.1 % -> g = {report(g, sg, 'm/s^2')}; "
      f"relative error of g = {100 * sg / g:.2f} % (the period contributes 2 x 1 %)")
rng = np.random.default_rng(5)
gs = 4 * np.pi**2 * rng.normal(l, sl, 200000) / rng.normal(Tp, sT, 200000)**2
print(f"Monte Carlo check: std(g)/g = {100 * gs.std() / gs.mean():.2f} %")
