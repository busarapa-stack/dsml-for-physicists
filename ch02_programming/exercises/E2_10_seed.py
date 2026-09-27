"""E2.10  What one seed can and cannot tell you.

Book answer: same seed -> identical mean; 20 seeds -> means scatter around 0 with
standard deviation about 1/sqrt(1000) = 0.03.
"""
import numpy as np


def mean_of_gaussians(rng, n=1000):
    return rng.normal(0.0, 1.0, n).mean()


a = mean_of_gaussians(np.random.default_rng(7))
b = mean_of_gaussians(np.random.default_rng(7))
print(f"seed 7 twice: {a:+.6f}, {b:+.6f} -> identical: {a == b}")

means = np.array([mean_of_gaussians(np.random.default_rng(s)) for s in range(20)])
print("20 seeds:", " ".join(f"{m:+.3f}" for m in means))
print(f"spread of the 20 means: std = {means.std(ddof=1):.4f}   "
      f"(theory 1/sqrt(1000) = {1 / np.sqrt(1000):.4f})")
print(f"smallest {means.min():+.3f}, largest {means.max():+.3f}: "
      "a single seed could have given any value in this range")
