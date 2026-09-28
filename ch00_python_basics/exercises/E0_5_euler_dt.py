"""E0.5  Error of the Euler free-fall time for three time steps."""
import math
g, h0 = 9.78, 20.0
t_exact = math.sqrt(2 * h0 / g)
for dt in [0.1, 0.01, 0.001]:
    t, y, v = 0.0, h0, 0.0
    while y > 0:
        y = y + v * dt
        v = v - g * dt
        t = t + dt
    print(f"dt = {dt:<6} t = {t:.4f} s  error = {t - t_exact:+.4f} s  error/dt = {(t - t_exact) / dt:.2f}")
