"""E6.10  Quasars as 'signal' (one-vs-rest), random forest without redshift, 200 expected quasars and
20,000 other objects in the sky area of interest: the threshold that maximises Z_A."""
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
__file__ = str(HERE / '06_sdss_colors.py')          # the listing's os.chdir uses __file__
src = (HERE / '06_sdss_colors.py').read_text()
exec(src[:src.index('\n# --- end of listings')].split('"""', 2)[2])
print(Path('data/raw/SOURCE.txt').read_text().strip())

feats = ['u_g', 'g_r', 'r_i', 'i_z', 'r']
Xtr, Xte, ytr, yte = train_test_split(df[feats].values, df['class'].to_numpy(), test_size=0.25,
                                      stratify=df['class'].to_numpy(), random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
p_qso = rf.predict_proba(Xte)[:, list(rf.classes_).index('QSO')]
yq = (yte == 'QSO').astype(int)
fpr, tpr, thr = roc_curve(yq, p_qso)
s_exp, b_exp = 200.0, 20000.0
S, B = s_exp * tpr, b_exp * fpr
ok = B >= 10.0
Z = np.zeros_like(S)
Z[ok] = np.sqrt(2 * ((S[ok] + B[ok]) * np.log1p(S[ok] / B[ok]) - S[ok]))
i = np.argmax(Z)
print(f"QSO-vs-rest AUC = {roc_auc_score(yq, p_qso):.3f}")
print(f"best threshold {thr[i]:.3f}: efficiency {tpr[i]:.3f}, contamination rate {fpr[i]:.4f}, "
      f"S = {S[i]:.0f}, B = {B[i]:.0f}, purity {S[i] / (S[i] + B[i]):.2f}, Z_A = {Z[i]:.1f}")
