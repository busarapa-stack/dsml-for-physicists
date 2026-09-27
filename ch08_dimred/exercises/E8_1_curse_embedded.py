"""E8.1  Max/min pairwise distance ratio for uniform data in D = 3, 10, 50, 100, and for a 2-D plane embedded in 100-D.
The exercise fixes no seed; five seeds are shown because the ratio depends on the single closest pair."""
import numpy as np
from scipy.spatial.distance import pdist

for seed in (2026, 0, 1, 2, 42):
    rng = np.random.default_rng(seed)
    r = {D: pdist(rng.random((200, D))) for D in (3, 10, 50, 100)}
    P = rng.random((200, 2))                          # 2-D points
    E = P @ rng.random((2, 100))                      # embedded in 100-D by a random 2 x 100 matrix
    d2, de = pdist(P), pdist(E)
    print(f"seed {seed:4d}: " + "  ".join(f"D={D}: {d.max() / d.min():6.1f}" for D, d in r.items())
          + f"  | plane in 100-D: {de.max() / de.min():7.0f}  (same points before embedding: {d2.max() / d2.min():6.0f})")
