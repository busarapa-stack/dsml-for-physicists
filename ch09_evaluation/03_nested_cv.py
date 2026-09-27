"""Ch.9  Optimistic bias of the best grid-search score vs nested CV, on pure-noise labels (20 data sets).

Book references: ssec:cv-select, code:nested-cv (line for line).  About 1 minute.
numpy and cross_val_score come from code:blocked-cv (01_blocked_cv.py); they are imported here directly.
"""
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score

os.chdir(Path(__file__).resolve().parent)

# --- code:nested-cv ----------------------------------------------------------
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

res = []
for seed in range(20):                           # 20 independent data sets
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 20))
    y = rng.integers(0, 2, 60)                   # labels are pure noise
    grid = {'svc__C': np.logspace(-2, 3, 6), 'svc__gamma': np.logspace(-4, 1, 6)}
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    outer = StratifiedKFold(5, shuffle=True, random_state=seed + 100)
    gs = GridSearchCV(make_pipeline(StandardScaler(), SVC()), grid,
                      cv=inner, scoring='roc_auc')
    gs.fit(X, y)
    nested = cross_val_score(gs, X, y, cv=outer, scoring='roc_auc').mean()
    res.append((gs.best_score_, nested))
print(np.mean(res, axis=0).round(3))             # [best inner score, nested]
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    r = np.array(res)
    print(f"best inner score > 0.5 in {np.sum(r[:, 0] > 0.5)} of 20 data sets; grid of {6 * 6} (C, gamma) pairs")
