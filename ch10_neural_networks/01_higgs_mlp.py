"""Ch.10  A two-layer network (MLP) for the Chapter 6 toy Higgs data, trained with PyTorch.

Book references: ssec:pytorch-mlp, ssec:train-loop, code:mlp and code:train (line for line)
Numbers quoted in the text: 4,610 parameters; AUC 0.979, accuracy 0.935, final loss 0.126;
logistic regression 0.703, random forest 0.975 (Chapter 6); without activations AUC 0.688.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import higgs_split, section, CODE

os.chdir(HERE)
Path('outputs').mkdir(exist_ok=True)
nb = higgs_split({})
X_train, X_val, y_train, y_val = nb['X_train'], nb['X_val'], nb['y_train'], nb['y_val']

# --- code:mlp ----------------------------------------------------------------
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class HiggsMLP(nn.Module):
    def __init__(self, n_in=4, hidden=64, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))      # raw scores (logits), no softmax
    def forward(self, x):
        return self.net(x)

scaler = StandardScaler().fit(X_train)         # X_train, X_val from Chapter 6
Xt = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
yt = torch.tensor(y_train)
Xv = torch.tensor(scaler.transform(X_val), dtype=torch.float32)

torch.manual_seed(0)
model = HiggsMLP()
print(sum(p.numel() for p in model.parameters()), "parameters")
# --- code:train --------------------------------------------------------------
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
history = []
for epoch in range(30):
    model.train()                                # training mode
    total = 0.0
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)            # 1. forward pass and loss
        optimizer.zero_grad()                    # 2. clear old gradients
        loss.backward()                          # 3. backpropagation
        optimizer.step()                         # 4. update parameters
        total += loss.item() * len(xb)
    history.append(total / len(Xt))

model.eval()                                     # evaluation mode
with torch.no_grad():                            # no gradient bookkeeping
    p_val = torch.softmax(model(Xv), dim=1)[:, 1].numpy()
print("AUC =", round(roc_auc_score(y_val, p_val), 4))
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    acc = np.mean((p_val > 0.5) == y_val)
    print(f"final training loss {history[-1]:.3f} (epoch 1: {history[0]:.3f}); accuracy at P > 0.5: {acc:.3f}")

    # the same network with the two ReLUs removed: a product of linear maps is linear
    torch.manual_seed(0)
    linear = nn.Sequential(nn.Linear(4, 64), nn.Linear(64, 64), nn.Linear(64, 2))
    opt = torch.optim.Adam(linear.parameters(), lr=1e-3)
    for epoch in range(30):
        linear.train()
        for xb, yb in loader:
            loss = loss_fn(linear(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    linear.eval()
    with torch.no_grad():
        p_lin = torch.softmax(linear(Xv), dim=1)[:, 1].numpy()
    print(f"no activation functions: AUC = {roc_auc_score(y_val, p_lin):.3f}")

    # Chapter 6 baselines on the same split (code:lr and code:rf)
    ch6 = dict(nb)
    ch6['__file__'] = str(CODE / 'ch06_classification' / '02_logreg_forest.py')
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):          # code:lr prints its coefficients
        exec(section(ch6['__file__'], '\n# --- code:lr', '\n# --- end of listings'), ch6)
    for name in ('lr_pipe', 'rf_pipe'):
        print(f"Chapter 6 {name}: AUC = {roc_auc_score(y_val, ch6[name].predict_proba(X_val)[:, 1]):.3f}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, 31), history, 'o-', ms=3)
    ax.set_xlabel('epoch'); ax.set_ylabel('training loss (cross entropy)')
    fig.tight_layout(); fig.savefig('outputs/ch10_higgs_loss.png', dpi=100)
    print("saved outputs/ch10_higgs_loss.png")
