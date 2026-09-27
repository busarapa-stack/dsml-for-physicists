"""Ch.3  Joining two tables safely: check keys, then compare row counts.

Book references: ssec:merge, tab:join, code:merge, review question 5 (10,000 x 10,000 -> 10,250)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent)

# --- code:merge -------------------------------------------------------------
metadata    = pd.read_csv('data/raw/galaxy10_metadata.csv', index_col='image_id')
image_paths = pd.read_csv('data/raw/galaxy10_image_index.csv', index_col='image_id')

assert metadata.index.is_unique, "duplicate image_id in metadata"
assert image_paths.index.is_unique, "duplicate image_id in image_paths"

combined = metadata.merge(image_paths,
                          left_index=True, right_index=True, how='inner')
print(len(metadata), len(image_paths), len(combined))

for how in ('inner', 'left', 'right', 'outer'):
    n = len(metadata.merge(image_paths, left_index=True, right_index=True, how=how))
    print(f"how='{how:5s}': {n} rows")
print("-> 20 galaxies have no image and 15 images have no metadata row; 'inner' drops both"
      " silently, 'outer' shows them")

# --- review question 5: 10,000 x 10,000 rows -> 10,250 rows -------------------
# rows of an inner join = sum over keys of n_A(key) * n_B(key).
# If A had unique keys, every row of B could match at most one row of A, so the result
# could never exceed len(B) = 10,000. Getting 10,250 therefore needs duplicates in BOTH.
rng = np.random.default_rng(5)
dup = rng.choice(10_000, 125, replace=False)             # 125 keys recorded twice ...
single = np.setdiff1d(np.arange(10_000), dup)[:9_750]
keys = np.concatenate([single, np.repeat(dup, 2)])      # 9,750 + 250 = 10,000 rows
A = pd.DataFrame({'key': keys, 'x': rng.normal(size=keys.size)})
B = pd.DataFrame({'key': keys, 'y': rng.normal(size=keys.size)})   # ... in both tables
print(f"\nA: {len(A)} rows, B: {len(B)} rows, 125 keys appear twice in each")
print(f"inner join: {len(A.merge(B, on='key', how='inner')):,} rows  (9,750 + 125 x 2 x 2)")
B1 = B.drop_duplicates('key')
print(f"with duplicates only in A (B made unique, {len(B1)} rows): "
      f"{len(A.merge(B1, on='key', how='inner')):,} rows -> never more than len(A)")
