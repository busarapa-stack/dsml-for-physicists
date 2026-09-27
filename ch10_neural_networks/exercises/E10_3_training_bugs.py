"""E10.3  Three bugs in the training loop of code:train, one at a time, same seed.

  bug 1  optimizer.zero_grad() removed      -> gradients accumulate over all previous steps
  bug 2  torch.softmax applied before CrossEntropyLoss (softmax twice)
  bug 3  no StandardScaler (raw GeV features)
Answer in the text: bug 1 AUC ~0.873, accuracy 0.797 (< 0.800 of 'always background');
bug 2 AUC 0.978 but final loss ~0.366 instead of 0.126; bug 3 AUC ~0.945.
About 1 minute.
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
HiggsMLP, y_train, y_val = nb['HiggsMLP'], nb['y_train'], nb['y_val']


def run(zero_grad=True, softmax_twice=False, scale=True):
    if scale:
        Xt, Xv = nb['Xt'], nb['Xv']
    else:
        Xt = torch.tensor(nb['X_train'], dtype=torch.float32)
        Xv = torch.tensor(nb['X_val'], dtype=torch.float32)
    yt = torch.tensor(y_train)
    torch.manual_seed(0)
    model = HiggsMLP()
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(30):
        model.train(); total = 0.0
        for xb, yb in loader:
            out = model(xb)
            if softmax_twice:
                out = torch.softmax(out, dim=1)                # BUG 2
            loss = loss_fn(out, yb)
            if zero_grad:
                optimizer.zero_grad()                          # BUG 1 when skipped
            loss.backward(); optimizer.step()
            total += loss.item() * len(xb)
        history.append(total / len(Xt))
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(Xv), dim=1)[:, 1].numpy()
    return history, p


print(f"accuracy of 'always background': {1 - y_val.mean():.3f}")
print(f"{'case':22s} loss ep1  loss ep30   AUC    acc")
for name, kw in [('correct loop', {}), ('no zero_grad', {'zero_grad': False}),
                 ('softmax twice', {'softmax_twice': True}), ('no scaling', {'scale': False})]:
    h, p = run(**kw)
    print(f"{name:22s} {h[0]:8.3f}  {h[-1]:9.3f}  {roc_auc_score(y_val, p):.3f}  {np.mean((p > 0.5) == y_val):.3f}")
