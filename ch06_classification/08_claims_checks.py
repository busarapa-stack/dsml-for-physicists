"""Ch.6  Numerical checks of statements in the text that have no listing of their own.

1. cross-entropy terms quoted in ssec:lr: -ln 0.01 and -ln 0.4
2. F1 for purity 0.99 and efficiency 0.05 (ssec:f1-imbalance)
3. AUC does not depend on the class proportions of the test set, purity does (ssec:roc-auc, ssec:roc-PI)
4. E6.1 arithmetic (false positives per day, S/sqrt(B), days to 5 sigma)
5. toy mass resolution quoted in ssec:higgs-features ("about 10 %")
"""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
for f in ('01_higgs_toy_data.py', '02_logreg_forest.py'):
    src = (HERE / f).read_text()
    exec(src[src.index('\n# --- code:'):src.index('\n# --- end of listings')])
from sklearn.metrics import precision_score, roc_auc_score

print(f"\n1. -ln 0.01 = {-np.log(0.01):.2f}, -ln 0.4 = {-np.log(0.4):.2f}")
P, R = 0.99, 0.05
print(f"2. F1(0.99, 0.05) = {2 * P * R / (P + R):.3f}, arithmetic mean {(P + R) / 2:.2f}")

score = rf_pipe.predict_proba(X_val)[:, 1]
cut = 0.892                                         # best Z_A threshold of 04_roc_significance.py
rng = np.random.default_rng(2026)
sig, bkg = np.where(y_val == 1)[0], np.where(y_val == 0)[0]
print("3. same classifier, validation set resampled to other signal fractions:")
for n_sig in (1200, 480, 48):                      # 20 %, ~9 %, ~1 % signal with all 4,800 background
    idx = np.concatenate([rng.choice(sig, n_sig, replace=False), bkg])
    yy, ss = y_val[idx], score[idx]
    print(f"   signal fraction {yy.mean():.3f}: AUC {roc_auc_score(yy, ss):.3f}, "
          f"purity at cut {cut} = {precision_score(yy, (ss >= cut).astype(int)):.2f}")

fp = 0.05 * 1e6; z_day = 100 / np.sqrt(fp)
print(f"4. E6.1: {fp:,.0f} false positives/day, fake:real = {fp / 100:.0f}:1, purity {100 / (100 + fp):.4f}, "
      f"S/sqrt(B) per day = {z_day:.3f}, days to 5 sigma = {(5 / z_day)**2:.0f}")
print(f"5. toy mass resolution 12/125 = {12 / 125:.3f}")
