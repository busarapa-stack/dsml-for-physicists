"""Ch.10  Learning-rate scan for the Higgs MLP and the 'dead ReLU' network at lr = 1.

Book references: ssec:loss-thermo, tab:lr-scan and the paragraph after it
The loop is code:train with only the learning rate changed (same seed, same data order).
Numbers quoted in the text (final training loss, validation AUC):
  1e-5: 0.327, 0.905   1e-4: 0.139, 0.977   1e-3: 0.126, 0.979
  1e-2: 0.129, 0.979   1e-1: 0.182, 0.966   1:    0.520, 0.500
  at lr = 1 all 64 units of the second hidden layer output 0 for every event; P ~ 0.22 for every event
About 2 minutes.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import higgs_split, listing

os.chdir(HERE)
nb = higgs_split({})
listing(nb, '01_higgs_mlp.py', 'mlp', '\n# --- code:train')      # HiggsMLP, Xt, yt, Xv
HiggsMLP, Xt, yt, Xv, y_val = nb['HiggsMLP'], nb['Xt'], nb['yt'], nb['Xv'], nb['y_val']


def fit(lr, epochs=30, seed=0):
    torch.manual_seed(seed)
    model = HiggsMLP()
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(xb)
        history.append(total / len(Xt))
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(Xv), dim=1)[:, 1].numpy()
    return model, history, p


print(f"{'lr':>7s}  loss(epoch 1, 5, 10, 30)          AUC")
for lr in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
    model, h, p = fit(lr)
    auc = roc_auc_score(y_val, p) if p.std() > 1e-6 else 0.5    # constant scores -> AUC 0.5
    print(f"{lr:7.0e}  {h[0]:.3f} {h[4]:.3f} {h[9]:.3f} {h[-1]:.3f}      {auc:.3f}")

# the network trained with lr = 1: which hidden units ever fire?
net = model.net
with torch.no_grad():
    h1 = torch.relu(net[0](Xt))
    h2 = torch.relu(net[2](h1))
    p_all = torch.softmax(model(Xt), dim=1)[:, 1]
print(f"lr = 1: units active for at least one training event: layer 1 {(h1.max(0).values > 0).sum().item()}/64, "
      f"layer 2 {(h2.max(0).values > 0).sum().item()}/64; P(signal) {p_all.min():.3f}-{p_all.max():.3f} for every event")
print(f"loss of a constant prediction 0.2: {-(0.8 * np.log(0.8) + 0.2 * np.log(0.2)):.3f}")
