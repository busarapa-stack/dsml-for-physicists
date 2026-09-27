"""Ch.12  Which seed matters? Random-forest seed vs data-split seed on the Chapter 6 toy Higgs data.

Book references: ssec:repro-practices, code:seed-spread (line for line); df from Chapter 6 code:higgs-data
Numbers quoted in the text: the same seeds twice give identical AUC; forest seed only: 0.9759-0.9765 (width 0.0006);
split seed only: 0.9770-0.9820 (width ~0.005, "almost ten times").
About 1 minute.
"""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
_src = (HERE.parent / 'ch06_classification' / '01_higgs_toy_data.py').read_text(encoding='utf-8')
_src = _src[_src.index('\n# --- code:higgs-data'):_src.index('\n# --- code:higgs-pipe')]
exec(_src)                                                   # make_higgs_toy, df (Chapter 6)

# --- code:seed-spread --------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

X, y = df[['m_bb', 'pT', 'dR', 'ETmiss']].values, df['label'].values

def run(split_seed, model_seed):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y,
                                              random_state=split_seed)
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10,
                                random_state=model_seed, n_jobs=-1).fit(X_tr, y_tr)
    return roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])

print("same seeds twice:", run(42, 0), run(42, 0))
auc_model = [run(42, s) for s in range(10)]        # only the forest's seed changes
auc_split = [run(s, 0) for s in range(10)]         # only the data split changes
print(f"model seed: mean {np.mean(auc_model):.4f}, range "
      f"{min(auc_model):.4f}-{max(auc_model):.4f}")
print(f"split seed: mean {np.mean(auc_split):.4f}, range "
      f"{min(auc_split):.4f}-{max(auc_split):.4f}")
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    w_m, w_s = np.ptp(auc_model), np.ptp(auc_split)
    print(f"widths: model seed {w_m:.4f}, split seed {w_s:.4f} (ratio {w_s / w_m:.1f})")
