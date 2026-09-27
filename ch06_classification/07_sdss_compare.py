"""Ch.6  Five classifiers on the three-class SDSS problem, macro-F1 by 5-fold CV;
then the same without the spectroscopic redshift (E6.5, notebox on data leakage).

Book references: ssec:sdss-cls, ssec:sdss-PI, E6.5, code:sdss-comp (reproduced line for line)
Uses the real table if present, otherwise the SYNTHETIC stand-in of 05_sdss_data.py.

    python 07_sdss_compare.py            # SVM trained on a random 10,000-row subset (book's advice)
    python 07_sdss_compare.py --full     # SVM on all 75,000 training rows (slow)
"""
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
src = (HERE / '06_sdss_colors.py').read_text()
exec(src[:src.index('\n# --- end of listings')].split('"""', 2)[2])      # code:sdss-cc (builds df)
FULL = '--full' in sys.argv
if not FULL:                              # the book: "subsample 10,000 training rows for SVM"
    from sklearn.svm import SVC as _SVC

    class SVC(_SVC):
        def fit(self, X, y, sample_weight=None):
            idx = np.random.default_rng(0).permutation(len(X))[:10_000]
            return super().fit(X[idx], y[idx])

# --- code:sdss-comp ----------------------------------------------------------
from sklearn.model_selection import cross_val_score, StratifiedKFold

features = ['u_g', 'g_r', 'r_i', 'i_z', 'r', 'redshift']
X = df[features].values
y = df['class'].to_numpy()          # a plain NumPy array of labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

classifiers = {
    'LogReg':     LogisticRegression(max_iter=1000),
    'KNN':        KNeighborsClassifier(n_neighbors=15),
    'SVM (RBF)':  SVC(kernel='rbf', random_state=42),
    'DecTree':    DecisionTreeClassifier(max_depth=8, random_state=42),
    'RandForest': RandomForestClassifier(n_estimators=200, random_state=42,
                                         n_jobs=-1),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
results = {}
for name, clf in classifiers.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_macro')
    results[name] = (scores.mean(), scores.std())
    print(f"{name:11s} macro-F1 = {scores.mean():.3f} +/- {scores.std():.3f}")
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    print(Path('data/raw/SOURCE.txt').read_text().strip(), "| SVM on", "all rows" if FULL else "10,000 rows")
    labels = ['GALAXY', 'STAR', 'QSO']

    def report(feats, tag):
        Xa = df[feats].values
        Xtr, Xte, ytr, yte = train_test_split(Xa, y, test_size=0.25, stratify=y, random_state=42)
        rf = Pipeline([('scaler', StandardScaler()),
                       ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]).fit(Xtr, ytr)
        yp = rf.predict(Xte)
        print(f"\n[{tag}] random forest, test set: macro-F1 = {f1_score(yte, yp, average='macro'):.3f}")
        cm = confusion_matrix(yte, yp, labels=labels)
        print("confusion matrix (rows true, columns predicted):", labels)
        print(cm)
        off = {f"{a}->{b}": cm[i, j] for i, a in enumerate(labels) for j, b in enumerate(labels) if i != j}
        print("largest confusions:", sorted(off.items(), key=lambda t: -t[1])[:3])
        imp = rf.named_steps['clf'].feature_importances_
        print("importance:", ", ".join(f"{f} {v:.2f}" for f, v in sorted(zip(feats, imp), key=lambda t: -t[1])))

    report(features, 'with redshift')
    no_z = [f for f in features if f != 'redshift']
    report(no_z, 'without redshift')
    from sklearn.model_selection import cross_val_score
    for name in ('LogReg', 'RandForest'):
        s = cross_val_score(Pipeline([('scaler', StandardScaler()), ('clf', classifiers[name])]),
                            df[no_z].values[:len(X)], y, cv=cv, scoring='f1_macro')
        print(f"without redshift, 5-fold CV on all rows: {name:10s} macro-F1 = {s.mean():.3f} +/- {s.std():.3f}")
