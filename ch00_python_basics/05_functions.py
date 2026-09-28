"""Ch.0  Functions with units in the docstring, checked against physics.

Book references: sec:py-functions, code:g-function, code:projectile-range, exercise E0.6
"""
import math
from math import pi

# --- code:g-function ---------------------------------------------------------
def g_from_pendulum(L_m, T_s):
    """Gravitational acceleration from a simple pendulum.

    L_m : length of the pendulum [m]
    T_s : period of one complete swing [s]
    returns g [m/s^2]  (small-amplitude formula T = 2 pi sqrt(L/g))
    """
    return 4 * pi**2 * L_m / T_s**2

print(f"g = {g_from_pendulum(0.80, 1.7971):.3f} m/s^2")

# --- code:projectile-range ---------------------------------------------------
def projectile_range(v0_ms, theta_deg, g=9.78):
    """Horizontal range on level ground [m]; v0 in m/s, angle in DEGREES."""
    theta_rad = math.radians(theta_deg)
    return v0_ms**2 * math.sin(2 * theta_rad) / g

for angle in [30, 45, 60]:
    print(f"theta = {angle} deg  R = {projectile_range(20.0, angle):.2f} m")

wrong = 20.0**2 * math.sin(2 * 45) / 9.78     # forgot to convert to radians
print(f"without radians: R = {wrong:.2f} m")

# --- end of listings ----------------------------------------------------------
best = max(range(0, 91), key=lambda a: projectile_range(20.0, a))
print("angle with the largest range:", best, "deg")
print("R(30) == R(60)?", math.isclose(projectile_range(20, 30), projectile_range(20, 60)))
