"""Ch.10  Uncertainty of the polymer classifier: MC dropout, a 5-network deep ensemble, and configuration spread.

Book references: ssec:mc-dropout, ssec:deep-ensemble, ssec:dl-uq-PI, code:mc-dropout (line for line), tab:nn-uq
Continues from code:polymer-nn (model, X_all, P, train_phase_classifier, crossing), run here from 04_polymer_classifier.py.
Numbers quoted in the text:
  tab:nn-uq (T, mean P, dropout SD, ensemble SD, config SD)
    1.0: 1.00 0.001 0.002 0.00   1.4: 0.40 0.060 0.084 0.46   1.8: 0.25 0.044 0.052 0.40
    2.4: 0.05 0.011 0.016 0.20   3.0: 0.01 0.002 0.006 0.07
  crossings of the five members 1.33-1.40 (training spread ~0.03); report 1.36 +/- 0.03;
  config-to-config spread ~5x the prediction uncertainty; ensemble SD above dropout SD at every T
About 1 minute.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import chain_data, listing

os.chdir(HERE)
Path('outputs').mkdir(exist_ok=True)
_nb = chain_data({'__name__': 'polymer', 'np': np, 'torch': torch, 'nn': nn})
listing(_nb, '04_polymer_classifier.py', 'polymer-nn')        # prints accuracy, T*, mean P, spread
globals().update({k: _nb[k] for k in ('T_lab', 'T_grid', 'dist', 'model', 'X_all', 'P',
                                      'train_phase_classifier', 'crossing')})

# --- code:mc-dropout ---------------------------------------------------------
def mc_dropout_predict(model, X, n_samples=50):
    model.train()                                   # keep dropout active
    with torch.no_grad():
        S = torch.stack([torch.softmax(model(X), dim=1)[:, 1]
                         for _ in range(n_samples)])
    model.eval()
    return S.mean(dim=0).numpy(), S.std(dim=0).numpy()

torch.manual_seed(1)
p_mc, s_mc = mc_dropout_predict(model, X_all)

ensemble = [train_phase_classifier(dist, T_lab, T_low=1.0, T_high=3.0, seed=s)
            for s in range(5)]                      # 5 independently initialised nets
P_ens = np.array([member[2] for member in ensemble])
T_stars = [crossing(p, T_lab, T_grid)[0] for p in P_ens]
print("T* of members:", np.round(T_stars, 2))
for T in T_grid:
    m = T_lab == T
    print(f"T = {T:.1f}  MC-dropout sd {s_mc[m].mean():.3f}  "
          f"ensemble sd {P_ens[:, m].std(axis=0).mean():.3f}  "
          f"config-to-config sd {P[m].std():.2f}")
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    Ts = np.array(T_stars)
    print(f"member crossings {Ts.min():.2f}-{Ts.max():.2f}; mean {Ts.mean():.2f} +/- {Ts.std(ddof=1):.2f} (SD of members)")
    sd_mc = np.array([s_mc[T_lab == t].mean() for t in T_grid])
    sd_en = np.array([P_ens[:, T_lab == t].std(axis=0).mean() for t in T_grid])
    sd_cf = np.array([P[T_lab == t].std() for t in T_grid])
    pbar = np.array([P[T_lab == t].mean() for t in T_grid])
    mid = (T_grid >= 1.2) & (T_grid <= 2.0)
    print(f"ensemble SD > dropout SD at {np.sum(sd_en > sd_mc)}/{len(T_grid)} temperatures; "
          f"exceptions: {[(float(t), round(float(a), 5), round(float(b), 5)) for t, a, b in zip(T_grid, sd_en, sd_mc) if a <= b]}")
    print(f"T = 1.2-2.0: config SD / ensemble SD = {(sd_cf[mid] / sd_en[mid]).min():.1f}-{(sd_cf[mid] / sd_en[mid]).max():.1f}, "
          f"config SD / dropout SD = {(sd_cf[mid] / sd_mc[mid]).min():.1f}-{(sd_cf[mid] / sd_mc[mid]).max():.1f}")
    print("\ntab:nn-uq")
    for t in (1.0, 1.4, 1.8, 2.4, 3.0):
        k = np.isclose(T_grid, t)
        print(f"  {t:.1f} & {pbar[k][0]:.2f} & {sd_mc[k][0]:.3f} & {sd_en[k][0]:.3f} & {sd_cf[k][0]:.2f}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    pm_mc = np.array([p_mc[T_lab == t].mean() for t in T_grid])
    pm_en = P_ens.mean(0); pm_en = np.array([pm_en[T_lab == t].mean() for t in T_grid])
    ax.fill_between(T_grid, pm_mc - sd_mc, pm_mc + sd_mc, alpha=0.3, label='MC dropout (mean SD)')
    ax.fill_between(T_grid, pm_en - sd_en, pm_en + sd_en, alpha=0.3, label='ensemble of 5 (mean SD)')
    ax.errorbar(T_grid, pbar, yerr=sd_cf, fmt='ko', ms=3, capsize=2, label='config-to-config SD')
    ax.axhline(0.5, color='grey', lw=0.5); ax.set_xlabel('T'); ax.set_ylabel(r'$\bar P$(globule)'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig('outputs/ch10_uncertainty.png', dpi=100)
    print("saved outputs/ch10_uncertainty.png")
