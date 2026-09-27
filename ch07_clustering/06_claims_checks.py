"""Ch.7  Checks of statements in the text that have no listing of their own.

  * curse of dimensionality: (d_max - d_min)/d_min for 500 uniform points, D = 2, 10, 100, 1000
  * review question 7: Delta G / k_B T = -ln(0.2/0.5)
"""
import numpy as np

SEED = 2026


def contrast(D, rng, n=500):
    P = rng.random((n, D))            # 500 points in the unit cube
    q = rng.random(D)                 # one more random query point
    d = np.linalg.norm(P - q, axis=1)
    return (d.max() - d.min()) / d.min()


rng = np.random.default_rng(SEED)
print("seed 2026 :", {D: round(contrast(D, rng), 2) for D in (2, 10, 100, 1000)}, "  (book: 33, 2.8, 0.44, 0.10)")
print("other seeds, 200 repeats each (median and 10-90 % range):")
for D in (2, 10, 100, 1000):
    r = np.array([contrast(D, np.random.default_rng(s)) for s in range(200)])
    print(f"  D = {D:4d}: median {np.median(r):6.2f}   range {np.percentile(r, 10):6.2f} - {np.percentile(r, 90):6.2f}")

print(f"\nQ7: Delta G / kT = -ln(0.2/0.5) = {-np.log(0.2 / 0.5):.2f}")
