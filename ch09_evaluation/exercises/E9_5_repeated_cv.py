"""E9.5  5 x 5 repeated stratified CV of KNN, SVM and random forest on the whole toy Higgs data set (same splits for all),
naive and Nadeau-Bengio corrected paired t-tests.  About 1-2 minutes."""
import os
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
warnings.filterwarnings('ignore')
_CH6 = HERE.parent / 'ch06_classification'
for _f in ('01_higgs_toy_data.py', '02_logreg_forest.py', '03_compare_five.py'):
    __file__ = str(_CH6 / _f)
    _s = (_CH6 / _f).read_text()
    exec(_s[_s.index('\n# --- code:'):_s.index('\n# --- end of listings')])
os.chdir(HERE)

X_all, y_all = df[features].to_numpy(), df['label'].to_numpy()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=2026)
auc = {}
for name in ('SVM (RBF)', 'RandForest', 'KNN'):
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', classifiers[name])])
    auc[name] = cross_val_score(pipe, X_all, y_all, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"{name:11s} mean AUC {auc[name].mean():.4f}")
J, ratio = 25, 1 / 4                                       # n_test / n_train for 5-fold CV
print(f"correction factor sqrt(1 + J n_test/n_train) = {np.sqrt(1 + J * ratio):.2f}")
for a, b in [('SVM (RBF)', 'RandForest'), ('SVM (RBF)', 'KNN'), ('RandForest', 'KNN')]:
    d = auc[a] - auc[b]
    t_naive = d.mean() / (d.std(ddof=1) / np.sqrt(J))
    t_nb = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1))
    print(f"{a} - {b}: {d.mean():+.4f} (sd {d.std(ddof=1):.4f}); naive t = {t_naive:.1f}, p = {2 * stats.t.sf(abs(t_naive), J - 1):.1g}; "
          f"corrected t = {t_nb:.1f}, p = {2 * stats.t.sf(abs(t_nb), J - 1):.1g}")
