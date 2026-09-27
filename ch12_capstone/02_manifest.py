"""Ch.12  A run manifest (versions, settings, SHA-256 of the data) and a check for duplicated rows across a split.

Book references: ssec:repro-manifest, code:manifest (line for line); continues from code:seed-spread (X, y, df)
Writes higgs_toy.csv and run_manifest.json into data/.
Numbers quoted in the text: clean split 0 overlapping rows; 300 rows recorded twice -> 121 test rows with a twin in training.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
(HERE / 'data').mkdir(exist_ok=True)
_src = (HERE.parent / 'ch06_classification' / '01_higgs_toy_data.py').read_text(encoding='utf-8')
exec(_src[_src.index('\n# --- code:higgs-data'):_src.index('\n# --- code:higgs-pipe')])
_s = (HERE / '01_seed_spread.py').read_text(encoding='utf-8')
_s = _s[_s.index('\n# --- code:seed-spread'):_s.index('\ndef run(')]
exec(_s)                                                     # imports and X, y of code:seed-spread
os.chdir(HERE / 'data')

# --- code:manifest -----------------------------------------------------------
import hashlib, json, platform, sys, datetime
import numpy, scipy, sklearn

def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()

def write_manifest(config, data_files, out='run_manifest.json'):
    """Record what is needed to rerun this analysis exactly."""
    manifest = {
        'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': {m.__name__: m.__version__ for m in (numpy, scipy, sklearn)},
        'config': config,                               # every seed and setting
        'data_sha256': {p: sha256_of(p) for p in data_files},
    }
    with open(out, 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest

def row_overlap(A, B, decimals=6):
    """Number of rows of B that also appear in A (duplicates across a split)."""
    key = lambda r: hashlib.md5(numpy.round(r, decimals).tobytes()).hexdigest()
    seen = {key(r) for r in A}
    return sum(key(r) in seen for r in B)

df.to_csv('higgs_toy.csv', index=False)                 # the exact data used
config = {'split_seed': 42, 'model_seed': 0, 'test_size': 0.3,
          'n_estimators': 100, 'min_samples_leaf': 10}
m = write_manifest(config, ['higgs_toy.csv'])
print(m['packages'], m['data_sha256']['higgs_toy.csv'][:16])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
print("overlap, clean split:", row_overlap(X_tr, X_te))
X_dup = np.vstack([X, X[:300]])                          # 300 rows recorded twice
y_dup = np.concatenate([y, y[:300]])
X_tr, X_te, _, _ = train_test_split(X_dup, y_dup, test_size=0.3, random_state=42)
print("overlap, data with duplicates:", row_overlap(X_tr, X_te))
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    X_dup = np.vstack([X, X[:300]]); y_dup = np.concatenate([y, y[:300]])
    X_tr, X_te, _, _ = train_test_split(X_dup, y_dup, test_size=0.3, random_state=42)
    print(f"expected twins in the test set if the 300 duplicates were split at random: "
          f"2 x 300 x 0.3 x 0.7 = {2 * 300 * 0.3 * 0.7:.0f}")
