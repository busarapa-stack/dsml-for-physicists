"""Ch.2  Loop versus vectorized NumPy, and broadcasting.

Book references: ssec:ndarray, code:vec-vs-loop, code:broadcast, code:broadcast-2d
The book says vectorized code can be "tens to hundreds of times" faster for large
arrays. The exact factor depends on the computer; this script measures it.
"""
import time

import numpy as np

SEED = 2026
rng = np.random.default_rng(SEED)

# --- code:vec-vs-loop -------------------------------------------------------
N = 1_000_000
m = rng.uniform(1.0, 2.0, N)          # kg
v = rng.normal(0.0, 1.0, N)           # m/s

t0 = time.perf_counter()
# loop: one particle at a time (avoid)
KE_total = 0.0
for i in range(N):
    KE_total += 0.5 * m[i] * v[i]**2
t1 = time.perf_counter()
KE_loop = KE_total

# vectorized: whole array at once
KE_total = np.sum(0.5 * m * v**2)
t2 = time.perf_counter()
KE_vec = KE_total

print(f"N = {N:,} particles")
print(f"loop       : KE = {KE_loop:.6e} J   time {t1 - t0:.3f} s")
print(f"vectorized : KE = {KE_vec:.6e} J   time {t2 - t1:.4f} s")
print(f"same result (relative difference {abs(KE_loop - KE_vec) / KE_vec:.1e}); "
      f"speed-up on this computer = {(t1 - t0) / (t2 - t1):.0f}x")

# 3D velocities, N x 3 (text after code:vec-vs-loop)
v3 = rng.normal(0.0, 1.0, (N, 3))
KE3 = np.sum(0.5 * m * np.sum(v3**2, axis=1))
print(f"3D: v has shape {v3.shape}; KE = {KE3:.6e} J "
      f"(equipartition check: KE/N = {KE3 / N:.3f} ~ 3/2 * <m> * 1 = {1.5 * m.mean():.3f})")

# --- code:broadcast ---------------------------------------------------------
x0, v0, a = 0.0, 5.0, -9.81
t = np.linspace(0.0, 10.0, 1000)
x = x0 + v0 * t + 0.5 * a * t**2
print(f"\nbroadcast 1D: scalars x0, v0, a with t{t.shape} -> x{x.shape}")

# --- code:broadcast-2d ------------------------------------------------------
v0 = 20.0
theta = np.deg2rad([30, 45, 60])    # shape (3,)
t     = np.linspace(0, 2, 200)      # shape (200,)
y = (v0 * np.sin(theta)[:, None] * t[None, :]
     - 0.5 * 9.81 * t[None, :]**2)
# y.shape == (3, 200): 3 angles x 200 time points
print(f"broadcast 2D: theta{theta.shape}[:, None] with t{t.shape}[None, :] -> y{y.shape}")
assert y.shape == (3, 200)
# the same numbers from an explicit double loop
y_loop = np.array([[v0 * np.sin(th) * tt - 0.5 * 9.81 * tt**2 for tt in t] for th in theta])
print(f"max |y_broadcast - y_loop| = {np.abs(y - y_loop).max():.1e}")
