"""E10.2  Network width and the role of the non-linearity (Higgs MLP, code:mlp + code:train).

Five networks: ReLU with hidden width 4, 16, 64, 256, and width 64 with tanh; same seed, same loop.
Answer in the text: AUC ~0.971 / 0.978 / 0.979 / 0.979, tanh 0.978; accuracy 0.920-0.935;
parameters 50 -> ~67k; width 4 starts highest (~0.60) and ends at 0.160, the others end at 0.123-0.136.
About 1.5 minutes.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from _shared import higgs_split, listing

os.chdir(HERE)
nb = higgs_split({})
listing(nb, '01_higgs_mlp.py', 'mlp', '\n# --- code:train')
Xt, yt, Xv, y_val = nb['Xt'], nb['yt'], nb['Xv'], nb['y_val']


def make(hidden, act):
    return nn.Sequential(nn.Linear(4, hidden), act(), nn.Linear(hidden, hidden), act(), nn.Linear(hidden, 2))


print(f"{'network':12s} {'params':>7s}  loss ep1  loss ep30   AUC    acc")
for name, hidden, act in [('ReLU 4', 4, nn.ReLU), ('ReLU 16', 16, nn.ReLU), ('ReLU 64', 64, nn.ReLU),
                          ('ReLU 256', 256, nn.ReLU), ('tanh 64', 64, nn.Tanh)]:
    torch.manual_seed(0)
    model = make(hidden, act)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(30):
        model.train(); total = 0.0
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(xb)
        history.append(total / len(Xt))
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(Xv), dim=1)[:, 1].numpy()
    n_par = sum(q.numel() for q in model.parameters())
    print(f"{name:12s} {n_par:7d}  {history[0]:8.3f}  {history[-1]:9.3f}  {roc_auc_score(y_val, p):.3f}  {np.mean((p > 0.5) == y_val):.3f}")
