"""E6.7  Stability of random-forest feature importance over 20 bootstrap resamples of the training set."""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
for f in ('01_higgs_toy_data.py', '02_logreg_forest.py'):
    src = (HERE / f).read_text()
    exec(src[src.index('\n# --- code:'):src.index('\n# --- end of listings')])

from sklearn.base import clone
rng = np.random.default_rng(2026)
imps = []
for b in range(20):
    idx = rng.integers(0, len(X_train), len(X_train))
    m = clone(rf_pipe).fit(X_train[idx], y_train[idx])
    imps.append(m.named_steps['clf'].feature_importances_)
imps = np.array(imps)
print("\nfeature   mean    std   (20 bootstrap resamples)")
for f, mu, sd in zip(features, imps.mean(0), imps.std(0, ddof=1)):
    print(f"{f:7s}  {mu:.3f}  {sd:.4f}")
d = imps[:, features.index('dR')] - imps[:, features.index('pT')]
print(f"dR - pT: mean {d.mean():.3f}, std {d.std(ddof=1):.3f}; dR above pT in {np.mean(d > 0) * 20:.0f} of 20 resamples")
