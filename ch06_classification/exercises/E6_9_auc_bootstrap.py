"""E6.9  Bootstrap confidence interval of the validation AUC of the random forest; AUCs of SVM and KNN."""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
for f in ('01_higgs_toy_data.py', '02_logreg_forest.py'):
    src = (HERE / f).read_text()
    exec(src[src.index('\n# --- code:'):src.index('\n# --- end of listings')])
from sklearn.metrics import roc_auc_score

score = rf_pipe.predict_proba(X_val)[:, 1]
rng = np.random.default_rng(2026)
aucs = []
for _ in range(1000):
    i = rng.integers(0, len(y_val), len(y_val))
    aucs.append(roc_auc_score(y_val[i], score[i]))
lo, hi = np.percentile(aucs, [2.5, 97.5])
print(f"\nrandom forest AUC = {roc_auc_score(y_val, score):.3f}, bootstrap 95 % CI [{lo:.3f}, {hi:.3f}] "
      f"(std {np.std(aucs):.4f})")
print("SVM 0.977 and KNN 0.973 (tab:cls-results) lie", "inside" if lo <= 0.973 and 0.977 <= hi else "outside", "this interval")
