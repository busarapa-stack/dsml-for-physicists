"""Ch.6  Five classifiers on the toy Higgs data; confusion matrix; the |m_bb - 125| feature.

Book references: ssec:choose-cls, tab:cls-results, ssec:confusion, tab:metrics
Listing reproduced line for line: code:cls-compare
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
for f in ('01_higgs_toy_data.py', '02_logreg_forest.py'):
    src = (HERE / f).read_text()
    exec(src[src.index('\n# --- code:'):src.index('\n# --- end of listings')])

# --- code:cls-compare --------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

classifiers = {
    'LogReg':     LogisticRegression(class_weight='balanced', random_state=42),
    'KNN':        KNeighborsClassifier(n_neighbors=25),
    'SVM (RBF)':  SVC(kernel='rbf', class_weight='balanced', random_state=42),
    'DecTree':    DecisionTreeClassifier(max_depth=8, class_weight='balanced',
                                         random_state=42),
    'RandForest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                         min_samples_leaf=10,
                                         class_weight='balanced',
                                         random_state=42, n_jobs=-1),
}
for name, clf in classifiers.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    pipe.fit(X_train, y_train)
    score = (pipe.decision_function(X_val) if name == 'SVM (RBF)'
             else pipe.predict_proba(X_val)[:, 1])
    y_pred = pipe.predict(X_val)
    print(f"{name:11s} acc={accuracy_score(y_val, y_pred):.3f} "
          f"F1={f1_score(y_val, y_pred):.3f} AUC={roc_auc_score(y_val, score):.3f}")
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    from sklearn.metrics import confusion_matrix
    import numpy as np
    print(f"always 'background': accuracy {1 - y_val.mean():.3f}")

    # confusion matrix of the random forest of code:rf at the default threshold 0.5
    tn, fp, fn, tp = confusion_matrix(y_val, rf_pipe.predict(X_val)).ravel()
    print(f"\nrandom forest (code:rf): TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")
    print(f"purity (precision) = {tp / (tp + fp):.3f}, efficiency (recall) = {tp / (tp + fn):.3f}, "
          f"specificity = {tn / (tn + fp):.3f}")
    P, R = 0.99, 0.05
    print(f"F1 for purity 0.99 and efficiency 0.05: {2 * P * R / (P + R):.3f} (arithmetic mean {(P + R) / 2:.2f})")

    # physics feature: distance from the Higgs mass
    X2_train, X2_val = X_train.copy(), X_val.copy()
    X2_train[:, 0] = np.abs(X2_train[:, 0] - 125.0)
    X2_val[:, 0] = np.abs(X2_val[:, 0] - 125.0)
    for name in ('LogReg', 'RandForest'):
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', classifiers[name])])
        pipe.fit(X2_train, y_train)
        print(f"with |m_bb - 125| instead of m_bb: {name:10s} AUC = "
              f"{roc_auc_score(y_val, pipe.predict_proba(X2_val)[:, 1]):.3f}")
