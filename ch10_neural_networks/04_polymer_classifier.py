"""Ch.10  Learning by extremes (Carrasquilla and Melko) on the polymer chain: where does P(globule) cross 0.5?

Book references: ssec:polymer-nn-train, ssec:nn-order-param, code:polymer-nn (line for line), tab:polymer-nn
Needs data/chain.npz from 03_polymer_chain.py (generated here if missing, about two minutes).
Numbers quoted in the text:
  validation accuracy 0.994; mean P = 1.00 for T <= 1.0, 0.70 / 0.40 / 0.29 at T = 1.2 / 1.4 / 1.6, 0 for T >= 3.2;
  T*_NN = 1.33; within-T SD of P 0.40-0.46 for T = 1.2-1.8; 86-90 % of configurations have P < 0.1 or P > 0.9
  tab:polymer-nn  (ends <=0.8/>=3.2, <=1.0/>=3.0, <=1.2/>=2.8)
     pair distances 1.25 / 1.33 / 1.43;  Rg^2 alone 1.33 / 1.55 / 1.56;  raw coordinates 0.93 / 1.32 / 1.41
  Rg^2 alone separates the training ends 99 %; raw coordinates validate 0.94-0.99 vs 0.98-0.99;
  raw coordinates with ends <= 0.8: mean P 0.21 at T = 1.0 but 0.41 at T = 1.2
About 1.5 minutes.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import chain_data

os.chdir(HERE)
_d = chain_data()
conf, ener, T_lab, T_grid, dist, rg2 = (_d[k] for k in ('conf', 'ener', 'T_lab', 'T_grid', 'dist', 'rg2'))

# --- code:polymer-nn ---------------------------------------------------------
def train_phase_classifier(F, T, T_low, T_high, seed=0, hidden=64, epochs=60):
    """Learning by extremes: label T <= T_low as globule (1), T >= T_high as coil (0)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = (T <= T_low) | (T >= T_high)
    y = (T <= T_low).astype(np.int64)
    mu, sd = F[train].mean(axis=0), F[train].std(axis=0) + 1e-9
    X_all = torch.tensor((F - mu) / sd, dtype=torch.float32)
    y_all = torch.tensor(y)
    idx = rng.permutation(np.where(train)[0])
    i_tr, i_va = idx[:int(0.8 * len(idx))], idx[int(0.8 * len(idx)):]
    model = nn.Sequential(nn.Linear(F.shape[1], hidden), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(hidden, 2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(i_tr)
        for b in range(0, len(perm), 64):
            bi = perm[b:b + 64]
            loss = loss_fn(model(X_all[bi]), y_all[bi])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        P = torch.softmax(model(X_all), dim=1)[:, 1].numpy()      # P(globule)
    val_acc = np.mean((P[i_va] > 0.5) == y[i_va])
    return model, X_all, P, val_acc

def crossing(P, T, T_grid):
    """Temperature where the mean P(globule) crosses 0.5 (linear interpolation)."""
    Pm = np.array([P[T == t].mean() for t in T_grid])
    k = np.where((Pm[:-1] - 0.5) * (Pm[1:] - 0.5) <= 0)[0][0]
    return T_grid[k] + (0.5 - Pm[k]) * (T_grid[k+1] - T_grid[k]) / (Pm[k+1] - Pm[k]), Pm

model, X_all, P, acc = train_phase_classifier(dist, T_lab, T_low=1.0, T_high=3.0)
T_star, Pm = crossing(P, T_lab, T_grid)
print("validation accuracy:", round(acc, 3), " T* =", round(T_star, 2))
print("mean P(globule):", Pm.round(2))
print("spread of P within each T:", np.array([P[T_lab == t].std() for t in T_grid]).round(2))
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    sd = np.array([P[T_lab == t].std() for t in T_grid])
    sure = np.array([np.mean((P[T_lab == t] < 0.1) | (P[T_lab == t] > 0.9)) for t in T_grid])
    band = (T_grid >= 1.2) & (T_grid <= 1.8)
    print(f"T = 1.2-1.8: within-T SD of P {sd[band].min():.2f}-{sd[band].max():.2f}; "
          f"fraction with P < 0.1 or P > 0.9: {sure[band].min():.2f}-{sure[band].max():.2f}")

    print("\ntab:polymer-nn  crossing of mean P = 0.5 (validation accuracy)")
    features = {'pair distances (435)': dist, 'Rg^2 alone': rg2[:, None],
                'raw coordinates (90)': conf.reshape(len(conf), -1)}
    ends = [(0.8, 3.2), (1.0, 3.0), (1.2, 2.8)]
    print(f"{'features':22s}" + ''.join(f"   <= {lo}, >= {hi}" for lo, hi in ends))
    for name, F in features.items():
        row = []
        for lo, hi in ends:
            _, _, Pq, a = train_phase_classifier(F, T_lab, lo, hi)
            ts, pm = crossing(Pq, T_lab, T_grid)
            row.append(f"   {ts:5.2f} ({a:.3f})  ")
            if name.startswith('raw') and lo == 0.8:
                raw_pm = pm
        print(f"{name:22s}" + ''.join(row))
    print(f"raw coordinates, ends <= 0.8 / >= 3.2: mean P at T = 1.0 {raw_pm[T_grid == 1.0][0]:.2f}, "
          f"at T = 1.2 {raw_pm[T_grid == 1.2][0]:.2f}")
    print(f"midpoint of <Rg^2>(T) from the tanh fit (03_polymer_chain.py): 1.90 +/- 0.04")
