"""E0.6  The range is largest at 45 degrees and equal for complementary angles."""
import math

def projectile_range(v0_ms, theta_deg, g=9.78):
    """Horizontal range on level ground [m]; v0 in m/s, angle in degrees."""
    return v0_ms**2 * math.sin(2 * math.radians(theta_deg)) / g

best = max(range(0, 91), key=lambda a: projectile_range(20.0, a))
print("largest range at", best, "deg:", round(projectile_range(20.0, best), 2), "m")
for a in [10, 20, 30, 40]:
    print(a, 90 - a, math.isclose(projectile_range(20.0, a), projectile_range(20.0, 90 - a)))
