"""Ch.12  Second-pass checks of statements that have no listing of their own.

  1  both listings run in book order as one notebook, continuing from Chapter 6 code:higgs-data
  2  SHA-256: changing one byte of the data file changes the whole hash                     (ssec:repro-manifest)
  3  121 test rows with a twin: expected 2 x 300 x 0.3 x 0.7 = 126 +/- 8.5; the forest is far more accurate on
     those rows ("predicted from memory")                                                  (ssec:repro-manifest)
  4  the 0.005 width from the data split vs the classifier difference tested in ssec:sig-testing (Chapter 9:
     AUC(SVM) - AUC(RF) = 0.0028 +/- 0.0010)                                               (ssec:repro-practices)
  5  the duplicate check takes "a few seconds"
About 2 minutes (item 4 reruns the Chapter 9 DeLong script).
"""
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
(HERE / 'data').mkdir(exist_ok=True)
os.chdir(HERE / 'data')

# 1 ------------------------------------------------------------------------------------------------
t0 = time.time()
nb = {'__name__': 'notebook'}
src = (HERE.parent / 'ch06_classification' / '01_higgs_toy_data.py').read_text(encoding='utf-8')
exec(src[src.index('\n# --- code:higgs-data'):src.index('\n# --- code:higgs-pipe')], nb)
for f, lab in [('01_seed_spread.py', 'seed-spread'), ('02_manifest.py', 'manifest')]:
    s = (HERE / f).read_text(encoding='utf-8')
    exec(s[s.index(f'\n# --- code:{lab}'):s.index('\n# --- end of listings')], nb)
print(f"1  both listings ran as one notebook in {time.time() - t0:.0f} s "
      f"(model-seed range {min(nb['auc_model']):.4f}-{max(nb['auc_model']):.4f}, "
      f"split-seed range {min(nb['auc_split']):.4f}-{max(nb['auc_split']):.4f})")

# 2 ------------------------------------------------------------------------------------------------
raw = bytearray(Path('higgs_toy.csv').read_bytes())
h1 = hashlib.sha256(raw).hexdigest()
i = len(raw) // 2
raw[i] = ord('7') if raw[i] != ord('7') else ord('8')          # change one digit in the middle of the file
h2 = hashlib.sha256(raw).hexdigest()
print(f"2  one byte changed: {sum(a != b for a, b in zip(h1, h2))} of 64 hex digits of SHA-256 differ")

# 3 and 5 ----------------------------------------------------------------------------------------
X, y, row_overlap = nb['X'], nb['y'], nb['row_overlap']
X_dup = np.vstack([X, X[:300]]); y_dup = np.concatenate([y, y[:300]])
idx = np.arange(len(y_dup))
tr, te = nb['train_test_split'](idx, test_size=0.3, random_state=42)
t1 = time.time()
n = row_overlap(X_dup[tr], X_dup[te])
dt = time.time() - t1
exp, sd = 2 * 300 * 0.3 * 0.7, np.sqrt(300 * 0.42 * 0.58)
print(f"3  test rows with a twin in training: {n} (expected {exp:.0f} +/- {sd:.1f})")
twin = np.isin(te % len(y), tr % len(y)) & ((te >= len(y)) | (te < 300))
for leaf in (10, 1):                                        # 10 as in code:seed-spread, 1 = fully grown trees
    rf = nb['RandomForestClassifier'](n_estimators=100, min_samples_leaf=leaf, random_state=0, n_jobs=-1).fit(X_dup[tr], y_dup[tr])
    ok = rf.predict(X_dup[te]) == y_dup[te]
    print(f"   forest min_samples_leaf={leaf}: accuracy on the {twin.sum()} test rows with a twin {ok[twin].mean():.3f}, "
          f"on the other test rows {ok[~twin].mean():.3f}")
print(f"5  duplicate check on {len(te):,} test rows against {len(tr):,} training rows: {dt:.2f} s")

# 4 ------------------------------------------------------------------------------------------------
out = subprocess.run([sys.executable, str(HERE.parent / 'ch09_evaluation' / '09_delong_calibration.py')],
                     capture_output=True, text=True).stdout
line = [l for l in out.splitlines() if l.startswith('AUC(SVM) - AUC(RF)')][0]
print(f"4  split-seed width {np.ptp(nb['auc_split']):.4f}; Chapter 9 DeLong: {line}")
