"""Ch.0  Conditions and loops: spring regimes, averaging, free fall step by step.

Book references: sec:py-control, code:spring-regime, code:mean-loop, code:euler-fall, exercise E0.5
"""
import math

# --- code:spring-regime ------------------------------------------------------
F_lim_N = 10.0      # limit of Hooke's law for this spring [N]
F_break_N = 25.0    # the spring breaks above this force [N]

for F_N in [4.0, 10.0, 18.0, 30.0]:
    if F_N <= F_lim_N:
        regime = "Hooke (linear)"
    elif F_N <= F_break_N:
        regime = "beyond Hooke (nonlinear)"
    else:
        regime = "broken"
    print(f"F = {F_N:5.1f} N  ->  {regime}")

# --- code:mean-loop ----------------------------------------------------------
t10_s = [17.66, 18.02, 17.76, 17.93, 17.93, 18.28, 17.97, 17.97, 17.90, 18.29]

total = 0.0
for t in t10_s:
    total = total + t / 10          # period of one swing [s]
T_mean_s = total / len(t10_s)
print(f"mean period T = {T_mean_s:.4f} s")

for k, t in enumerate(t10_s, start=1):
    if k <= 3:
        print(f"trial {k}: T = {t / 10:.3f} s")

# --- code:euler-fall ---------------------------------------------------------
g = 9.78            # [m/s^2]
h0 = 20.0           # drop height [m]
dt = 0.01           # time step [s]

t, y, v = 0.0, h0, 0.0
while y > 0:
    y = y + v * dt          # position from the old velocity
    v = v - g * dt          # velocity from the acceleration
    t = t + dt

t_exact = math.sqrt(2 * h0 / g)
print(f"Euler : t = {t:.3f} s")
print(f"exact : t = {t_exact:.3f} s")

# --- end of listings ----------------------------------------------------------
for dt in [0.1, 0.01, 0.001]:
    t, y, v = 0.0, h0, 0.0
    while y > 0:
        y = y + v * dt
        v = v - g * dt
        t = t + dt
    print(f"dt = {dt:<6} t = {t:.4f} s  error = {t - t_exact:+.4f} s")
