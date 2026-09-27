"""E6.4  ROC curve, Z_A versus threshold, and purity/efficiency versus threshold (expected real-data counts)."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
src = (HERE / '04_roc_significance.py').read_text()
exec(src[src.index('for f in'):src.index("if __name__ == '__main__':")])

purity = np.where(S + B > 0, S / np.where(S + B > 0, S + B, 1), np.nan)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], 'k:'); ax[0].set_xlabel('background efficiency (FPR)')
ax[0].set_ylabel('signal efficiency (TPR)'); ax[0].set_title(f'ROC, AUC = {auc:.3f}')
sel = thresholds <= 1
ax[1].plot(thresholds[sel], Z_A[sel]); ax[1].axvline(thresholds[i_opt], ls=':')
ax[1].set_xlabel('threshold'); ax[1].set_ylabel(r'$Z_A$ (50 signal, 5000 background expected)')
ax[2].plot(thresholds[sel], purity[sel], label='purity (expected real data)')
ax[2].plot(thresholds[sel], tpr[sel], label='signal efficiency'); ax[2].axvline(thresholds[i_opt], ls=':')
ax[2].set_xlabel('threshold'); ax[2].legend()
fig.tight_layout(); fig.savefig('outputs/E6_4_roc_threshold.png', dpi=120)
print(f"\nbest threshold {thresholds[i_opt]:.3f}: eff_S {tpr[i_opt]:.3f}, eff_B {fpr[i_opt]:.4f}, "
      f"purity {purity[i_opt]:.3f}, Z_A {Z_A[i_opt]:.2f}")
print("the s_exp = 500 and 2000 cases are printed by 04_roc_significance.py")
print("saved outputs/E6_4_roc_threshold.png")
