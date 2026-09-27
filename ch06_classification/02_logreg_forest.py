"""Ch.6  Logistic regression and random forest pipelines; feature importance.

Book references: ssec:lr, ssec:dt-rf, tab:rf-imp, ssec:f1-imbalance (predict_proba notebox)
Listings reproduced line for line: code:lr, code:rf
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
src = (HERE / '01_higgs_toy_data.py').read_text()
exec(src[src.index('\n# --- code:higgs-data'):src.index('\n# --- end of listings')])

# --- code:lr -----------------------------------------------------------------
from sklearn.linear_model import LogisticRegression

lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(C=1.0, class_weight='balanced',
                                  random_state=42)),
])
lr_pipe.fit(X_train, y_train)
y_proba_lr = lr_pipe.predict_proba(X_val)[:, 1]
print(lr_pipe.named_steps['clf'].coef_)    # one weight per feature
# --- code:rf -----------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

rf_pipe = Pipeline([
    ('scaler', StandardScaler()),      # harmless for trees; kept for uniformity
    ('clf',    RandomForestClassifier(n_estimators=200, max_depth=10,
                                      min_samples_leaf=10,
                                      class_weight='balanced',
                                      random_state=42, n_jobs=-1)),
])
rf_pipe.fit(X_train, y_train)
importance = rf_pipe.named_steps['clf'].feature_importances_
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    perm = permutation_importance(rf_pipe, X_val, y_val, scoring='roc_auc',
                                  n_repeats=10, random_state=42, n_jobs=-1)
    print("\nfeature   impurity   permutation (drop in validation AUC)")
    for f, a, b in sorted(zip(features, importance, perm.importances_mean), key=lambda t: -t[1]):
        print(f"{f:7s}   {a:8.3f}   {b:8.3f}")
    print(f"sums: impurity {importance.sum():.2f}, permutation {perm.importances_mean.sum():.2f}")
    p_rf = rf_pipe.predict_proba(X_val)[:, 1]
    print(f"\nmean predict_proba of the forest on the validation set = {p_rf.mean():.3f} "
          f"(true signal fraction there {y_val.mean():.3f})")
    print(f"logistic regression validation AUC = {roc_auc_score(y_val, y_proba_lr):.3f}")
