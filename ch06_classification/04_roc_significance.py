"""Ch.6  ROC curve, AUC and the threshold that maximises the discovery significance Z_A.

Book references: ssec:roc-auc, ssec:roc-PI, E6.4, review question 6
Listing reproduced line for line: code:roc-sb
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
for f in ('01_higgs_toy_data.py', '02_logreg_forest.py'):
    src = (HERE / f).read_text()
    exec(src[src.index('\n# --- code:'):src.index('\n# --- end of listings')])

# --- code:roc-sb -------------------------------------------------------------
from sklearn.metrics import roc_curve, roc_auc_score

y_score = rf_pipe.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_score)
auc = roc_auc_score(y_val, y_score)

# expected yields in real data after preselection (NOT validation-set counts)
s_exp, b_exp = 50.0, 5000.0
S = s_exp * tpr                         # expected signal passing each cut
B = b_exp * fpr                         # expected background passing each cut

ok = B >= 10.0                          # avoid cuts that leave almost no background
Z_A = np.zeros_like(S)
Z_A[ok] = np.sqrt(2 * ((S[ok] + B[ok]) * np.log1p(S[ok] / B[ok]) - S[ok]))
i_opt = np.argmax(Z_A)
print(f"AUC = {auc:.3f}")
print(f"best cut {thresholds[i_opt]:.3f}: eff_S = {tpr[i_opt]:.3f}, "
      f"eff_B = {fpr[i_opt]:.4f}, S = {S[i_opt]:.1f}, B = {B[i_opt]:.1f}, "
      f"Z = {Z_A[i_opt]:.2f}")
# --- end of listings ---------------------------------------------------------


def z_asimov(S, B):
    return np.sqrt(2 * ((S + B) * np.log1p(S / B) - S))


if __name__ == '__main__':
    print(f"\nno cut: S/sqrt(B) = {s_exp / np.sqrt(b_exp):.2f}, Z_A = {z_asimov(s_exp, b_exp):.2f}")
    print(f"gain in significance from the classifier: x{Z_A[i_opt] / z_asimov(s_exp, b_exp):.1f}")
    purity_real = S[i_opt] / (S[i_opt] + B[i_opt])
    n_s_val, n_b_val = y_val.sum(), (1 - y_val).sum()
    purity_val = tpr[i_opt] * n_s_val / (tpr[i_opt] * n_s_val + fpr[i_opt] * n_b_val)
    print(f"purity at the best cut: validation set {purity_val:.3f}, expected real data {purity_real:.3f}; "
          f"S/B = {S[i_opt] / B[i_opt]:.2f}")

    # S/sqrt(B) instead of Z_A
    z_simple = np.zeros_like(S)
    z_simple[ok] = S[ok] / np.sqrt(B[ok])
    j = np.argmax(z_simple)
    print(f"maximising S/sqrt(B) instead: cut {thresholds[j]:.3f}, B = {B[j]:.1f}, S/sqrt(B) = {z_simple[j]:.2f}, "
          f"but Z_A there = {Z_A[j]:.2f} (< {Z_A[i_opt]:.2f})")

    # the wrong way: counts of the validation set itself
    Sv, Bv = n_s_val * tpr, n_b_val * fpr
    okv = Bv > 0
    zv = np.where(okv, Sv / np.sqrt(np.where(okv, Bv, 1)), 0)
    k = np.argmax(zv)
    print(f"S/sqrt(B) with validation-set counts (meaningless): max {zv[k]:.0f} at B = {Bv[k]:.0f} event(s)")

    # review question 6: purity when the true signal fraction is 1 %
    f = 0.01
    print(f"purity at f = 1 %: {tpr[i_opt] * f / (tpr[i_opt] * f + fpr[i_opt] * (1 - f)):.2f}")

    # E6.4: s_exp = 500
    for s_e in (50.0, 500.0, 2000.0):
        S2, B2 = s_e * tpr, b_exp * fpr
        Z2 = np.zeros_like(S2); Z2[ok] = z_asimov(S2[ok], B2[ok])
        m = np.argmax(Z2)
        print(f"s_exp = {s_e:5.0f}: best cut {thresholds[m]:.3f}, eff_S = {tpr[m]:.3f}, eff_B = {fpr[m]:.4f}, "
              f"S = {S2[m]:.0f}, B = {B2[m]:.0f}, S/B = {S2[m] / B2[m]:.2f}, Z_A = {Z2[m]:.1f}")
