"""Ch.9  Paired comparison of classifiers (DeLong test, paired bootstrap, McNemar) and calibration of the
balanced random forest (Brier, ECE, isotonic), plus the prior-shift correction to a 1 % signal fraction.

Book references: ssec:sig-testing, ssec:calibration, code:delong, code:calibration (line for line)
Continues from the Chapter 6 listings (toy Higgs data, classifiers, rf_pipe), executed from ../ch06_classification.
"""
import os
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
warnings.filterwarnings('ignore')
_CH6 = HERE.parent / 'ch06_classification'
for _f in ('01_higgs_toy_data.py', '02_logreg_forest.py', '03_compare_five.py'):
    __file__ = str(_CH6 / _f)                                  # Chapter 6 listings chdir via __file__
    _s = (_CH6 / _f).read_text()
    exec(_s[_s.index('\n# --- code:'):_s.index('\n# --- end of listings')])
__file__ = str(HERE / '09_delong_calibration.py')
os.chdir(HERE)

# --- code:delong -------------------------------------------------------------
from scipy import stats

def delong(y, s1, s2):
    """Paired test for AUC(s1) - AUC(s2) on the same events."""
    pos, neg = y == 1, y == 0
    def components(s):
        sp, sn = s[pos], s[neg]
        psi = (sp[:, None] > sn[None, :]) + 0.5 * (sp[:, None] == sn[None, :])
        return psi.mean(), psi.mean(axis=1), psi.mean(axis=0)
    a1, v10_1, v01_1 = components(s1)
    a2, v10_2, v01_2 = components(s2)
    S10 = np.cov(np.vstack([v10_1, v10_2]))           # over signal events
    S01 = np.cov(np.vstack([v01_1, v01_2]))           # over background events
    S = S10 / pos.sum() + S01 / neg.sum()
    diff = a1 - a2
    se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])     # covariance term matters
    return diff, se, 2 * stats.norm.sf(abs(diff) / se)

score = {}                                        # classifiers: Chapter 6 listing
for name in ['KNN', 'SVM (RBF)', 'RandForest']:
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', classifiers[name])])
    pipe.fit(X_train, y_train)
    score[name] = (pipe.decision_function(X_val) if name == 'SVM (RBF)'
                   else pipe.predict_proba(X_val)[:, 1])
d, se, p = delong(y_val, score['SVM (RBF)'], score['RandForest'])
print(f"AUC(SVM) - AUC(RF) = {d:.4f} +/- {se:.4f}, p = {p:.3f}")
# --- code:calibration --------------------------------------------------------
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
               for b in range(bins) if np.any(idx == b))

rf_cal = CalibratedClassifierCV(rf_pipe, method='isotonic', cv=5)
for name, model in [('balanced RF', rf_pipe), ('+ isotonic', rf_cal)]:
    model.fit(X_train, y_train)
    p = model.predict_proba(X_val)[:, 1]
    frac_pos, mean_p = calibration_curve(y_val, p, n_bins=10)
    print(name, "Brier", round(brier_score_loss(y_val, p), 4),
          "ECE", round(ece(y_val, p), 4))
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score
    pos, neg = y_val == 1, y_val == 0

    def auc_se(s):
        psi = (s[pos][:, None] > s[neg][None, :]) + 0.5 * (s[pos][:, None] == s[neg][None, :])
        return np.sqrt(psi.mean(1).var(ddof=1) / pos.sum() + psi.mean(0).var(ddof=1) / neg.sum())

    print()
    for n, s in score.items():
        print(f"{n:11s} AUC {roc_auc_score(y_val, s):.4f}  SE {auc_se(s):.4f}")
    print(f"unpaired SE of the difference SVM - RF: {np.hypot(auc_se(score['SVM (RBF)']), auc_se(score['RandForest'])):.4f}")
    d2, se2, p2 = delong(y_val, score['RandForest'], score['KNN'])
    print(f"AUC(RF) - AUC(KNN) = {d2:.4f} +/- {se2:.4f}, p = {p2:.2f}")
    r = np.random.default_rng(0); D = []
    for _ in range(1000):
        i = r.integers(0, len(y_val), len(y_val))
        D.append(roc_auc_score(y_val[i], score['SVM (RBF)'][i]) - roc_auc_score(y_val[i], score['RandForest'][i]))
    print("paired bootstrap 95 % CI of AUC(SVM) - AUC(RF):", np.percentile(D, [2.5, 97.5]).round(4))

    pred = {}
    for n in ('KNN', 'RandForest'):
        pred[n] = Pipeline([('scaler', StandardScaler()), ('clf', classifiers[n])]).fit(X_train, y_train).predict(X_val)
    ok_rf, ok_knn = pred['RandForest'] == y_val, pred['KNN'] == y_val
    n01, n10 = int((ok_rf & ~ok_knn).sum()), int((~ok_rf & ok_knn).sum())
    chi = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    print(f"McNemar RF vs KNN: n01 = {n01}, n10 = {n10}, p = {stats.chi2.sf(chi, 1):.3f}; accuracy RF {ok_rf.mean():.3f}, KNN {ok_knn.mean():.3f}")

    p_raw = rf_pipe.fit(X_train, y_train).predict_proba(X_val)[:, 1]
    for lo, hi in [(0.4, 0.5), (0.5, 0.6)]:
        m = (p_raw >= lo) & (p_raw < hi)
        print(f"balanced RF, scores {lo}-{hi}: mean score {p_raw[m].mean():.2f}, signal fraction {y_val[m].mean():.2f}")
    p_cal = rf_cal.predict_proba(X_val)[:, 1]
    print(f"AUC after isotonic calibration {roc_auc_score(y_val, p_cal):.4f}")

    def prior_shift(p, pi_new, pi_train=0.2):
        a = pi_new / pi_train * p
        return a / (a + (1 - pi_new) / (1 - pi_train) * (1 - p))

    print(f"events with calibrated p > 0.5: {100 * (p_cal > 0.5).mean():.1f} %; after shifting to a 1 % signal fraction: "
          f"{100 * (prior_shift(p_cal, 0.01) > 0.5).mean():.1f} %")
