"""E6.3  Training and prediction time, and 5-fold stratified CV of F1, for the five classifiers."""
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
for f in ('01_higgs_toy_data.py', '02_logreg_forest.py', '03_compare_five.py'):
    src = (HERE / f).read_text()
    exec(src[src.index('\n# --- code:'):src.index('\n# --- end of listings')])
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print("\nmethod       fit (s)  predict (s)   CV F1 (mean +/- std)")
for name, clf in classifiers.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    t0 = time.perf_counter(); pipe.fit(X_train, y_train); t1 = time.perf_counter()
    pipe.predict(X_val); t2 = time.perf_counter()
    s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1')
    print(f"{name:11s}  {t1 - t0:7.3f}  {t2 - t1:9.3f}     {s.mean():.3f} +/- {s.std():.3f}")

X2_train = X_train.copy(); X2_train[:, 0] = np.abs(X2_train[:, 0] - 125.0)
s = cross_val_score(Pipeline([('scaler', StandardScaler()), ('clf', classifiers['LogReg'])]),
                    X2_train, y_train, cv=cv, scoring='f1')
print(f"LogReg with |m_bb - 125|: CV F1 {s.mean():.3f} +/- {s.std():.3f}")
print("(times depend on the computer; compare their ratios)")
