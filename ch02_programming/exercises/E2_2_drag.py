"""E2.2  Projectile with linear air drag, F = -b m v. Landing point with events.

Book answer: range about 34.1 m for v0 = 20 m/s, 45 deg, b = 0.1 1/s;
setting b = 0 must give back 40.79 m.
"""
import numpy as np
from scipy.constants import g
from scipy.integrate import solve_ivp


def projectile_drag_rhs(t, y, b):
    x, yy, vx, vy = y
    return [vx, vy, -b * vx, -g - b * vy]


def hit_ground(t, y, b):
    return y[1]


hit_ground.terminal = True        # stop the integration
hit_ground.direction = -1         # only when y goes from + to -


def landing_range(v0=20.0, theta_deg=45.0, b=0.1):
    th = np.deg2rad(theta_deg)
    y0 = [0.0, 0.0, v0 * np.cos(th), v0 * np.sin(th)]
    sol = solve_ivp(projectile_drag_rhs, (0, 10), y0, args=(b,), events=hit_ground,
                    rtol=1e-10, atol=1e-12)
    # the event at t = 0 (y = 0 while rising) is not counted because direction = -1
    t_land = sol.t_events[0][0]
    return sol.y_events[0][0][0], t_land


for b in (0.1, 0.0):
    R, T = landing_range(b=b)
    print(f"b = {b:3.1f} 1/s : range = {R:.2f} m, flight time = {T:.3f} s")
print(f"no-drag formula v0^2/g = {400 / g:.2f} m")

# which angle gives the longest range with drag? (it is below 45 deg)
angles = np.arange(30, 50.5, 0.5)
R = [landing_range(theta_deg=a)[0] for a in angles]
print(f"with b = 0.1 the best angle is {angles[int(np.argmax(R))]:.1f} deg "
      f"(range {max(R):.2f} m), not 45 deg")
