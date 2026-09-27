"""E6.6  SMOTE versus class weights on a strongly imbalanced toy Higgs set (400 signal, 40,000 background).
Needs imbalanced-learn (pip install imbalanced-learn)."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
src = (HERE / '01_higgs_toy_data.py').read_text()
exec(src[src.index('\n# --- code:higgs-data'):src.index('\n# --- code:higgs-pipe')])   # make_higgs_toy

df = make_higgs_toy(n_sig=400, n_bkg=40000)
features = ['m_bb', 'pT', 'dR', 'ETmiss']
X, y = df[features].values, df['label'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
rf = dict(n_estimators=200, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1)

models = {
    'no correction': Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(**rf))]),
    'class_weight':  Pipeline([('scaler', StandardScaler()),
                               ('clf', RandomForestClassifier(class_weight='balanced', **rf))]),
    'SMOTE':         ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=42)),
                                  ('clf', RandomForestClassifier(**rf))]),
}
print("model           AUC    efficiency  purity   (threshold 0.5, validation set)")
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict_proba(X_val)[:, 1]
    yp = (p >= 0.5).astype(int)
    print(f"{name:14s}  {roc_auc_score(y_val, p):.3f}   {recall_score(y_val, yp):.3f}      "
          f"{precision_score(y_val, yp, zero_division=0):.3f}")

# synthetic SMOTE signal events vs real signal in the dR - pT plane
Xs = StandardScaler().fit(X_train)
Xr, yr = SMOTE(random_state=42).fit_resample(Xs.transform(X_train), y_train)
synth = Xs.inverse_transform(Xr[len(X_train):])            # SMOTE appends new samples at the end
real = X_train[y_train == 1]
rel = lambda A: A[:, 2] * A[:, 1] / (2 * A[:, 0])           # dR * pT / (2 m_bb), about 1 for real signal
print(f"\n{len(synth)} synthetic signal events; dR*pT/(2 m_bb): real signal median {np.median(rel(real)):.2f}, "
      f"IQR {np.subtract(*np.percentile(rel(real), [75, 25])):.2f}; synthetic median {np.median(rel(synth)):.2f}, "
      f"IQR {np.subtract(*np.percentile(rel(synth), [75, 25])):.2f}")
lo, hi = np.percentile(rel(real), [1, 99])
print(f"fraction outside the 1-99 % band of the real signal [{lo:.2f}, {hi:.2f}]: real 0.02 by construction, "
      f"synthetic {np.mean((rel(synth) < lo) | (rel(synth) > hi)):.3f}")
print("-> SMOTE joins close neighbours, so here the curved dR-pT relation survives almost intact;"
      " the danger grows when the minority class is sparse or the relation bends sharply")
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(synth[:3000, 1], synth[:3000, 2], s=3, alpha=0.3, label='SMOTE synthetic')
ax.scatter(real[:, 1], real[:, 2], s=6, c='k', label='real signal')
pt = np.linspace(60, 600, 200); ax.plot(pt, 250 / pt, 'r--', label=r'$2 m_H / p_T$')
ax.set_xlabel(r'$p_T$ (GeV)'); ax.set_ylabel(r'$\Delta R$'); ax.set_ylim(0, 4.2); ax.legend()
fig.tight_layout(); fig.savefig('outputs/E6_6_smote_dR_pT.png', dpi=150)
print("saved outputs/E6_6_smote_dR_pT.png")
