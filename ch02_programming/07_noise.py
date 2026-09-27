"""Ch.2  Adding Gaussian noise with a fixed random seed (scipy.stats + Generator).

Book references: ssec:noise, code:noise
"""
import numpy as np
from scipy.constants import g
from scipy.stats import norm


def projectile(v0, theta_deg, n=1000):
    theta = np.deg2rad(theta_deg)
    T = 2.0 * v0 * np.sin(theta) / g
    t = np.linspace(0.0, T, n)
    return t, v0 * np.cos(theta) * t, v0 * np.sin(theta) * t - 0.5 * g * t**2


rng = np.random.default_rng(seed=42)
sigma_y = 0.02                        # m

t, x, y_clean = projectile(v0=20.0, theta_deg=45)
noise = norm.rvs(scale=sigma_y, size=y_clean.size, random_state=rng)
y_observed = y_clean + noise

print(f"{y_clean.size} points, sigma_y = {sigma_y} m")
print(f"sample std of the noise = {noise.std(ddof=1):.5f} m (expected {sigma_y})")
print(f"first three noise values: {np.round(noise[:3], 5)}")

# same seed -> identical noise; this is what makes the analysis reproducible
rng2 = np.random.default_rng(seed=42)
noise2 = norm.rvs(scale=sigma_y, size=y_clean.size, random_state=rng2)
print(f"re-created with seed 42 -> identical: {np.array_equal(noise, noise2)}")
