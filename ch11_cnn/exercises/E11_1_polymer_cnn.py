"""E11.1  A small CNN on the 30x30 distance matrices of the Chapter 10 polymer chain (learning by extremes).

Uses the Chapter 10 ensemble (code:chain-gen, ../ch10_neural_networks/data/chain.npz; generated if missing)
and train_phase_classifier / crossing of code:polymer-nn with the MLP replaced by a 3-layer CNN.
Answer in the text: 5,954 parameters vs 32,194 of the MLP; validation 0.978-0.997; crossings about 1.6, 1.3, 1.7
for ends (1.0, 3.0), (0.8, 3.2), (1.2, 2.8), seeds 1.47-1.60 for the first; all ends and seeds about 1.2-1.8.
The chain ensemble itself depends on the computer's rounding (see Chapter 10 README), so expect shifts of ~0.1.
About 3 minutes.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CH10 = Path(__file__).resolve().parents[2] / 'ch10_neural_networks'
sys.path.insert(0, str(CH10))
from _shared import chain_data, section as section10

nb = chain_data({'np': np, 'torch': torch, 'nn': nn})
conf, T_lab, T_grid = nb['conf'], nb['T_lab'], nb['T_grid']
D = np.linalg.norm(conf[:, :, None, :] - conf[:, None, :, :], axis=3).astype(np.float32)   # (6000, 30, 30)


class MapCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 30 -> 15
                               nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 15 -> 7
                               nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.h = nn.Linear(32, 2)

    def forward(self, x):
        return self.h(self.f(x.view(-1, 1, 30, 30)).mean(dim=(2, 3)))


src = section10(CH10 / '04_polymer_classifier.py', '\n# --- code:polymer-nn', 'model, X_all, P, acc =')
old = """    model = nn.Sequential(nn.Linear(F.shape[1], hidden), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(hidden, 2))"""
assert old in src
nb['MapCNN'] = MapCNN
exec(src.replace(old, "    model = MapCNN()"), nb)
n_mlp = 435 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2
print(f"parameters: CNN {sum(p.numel() for p in MapCNN().parameters())}, MLP of code:polymer-nn {n_mlp}")

F = D.reshape(len(D), -1)
allT = []
for lo, hi in [(1.0, 3.0), (0.8, 3.2), (1.2, 2.8)]:
    ts, accs = [], []
    for s in range(3):
        _, _, P, a = nb['train_phase_classifier'](F, T_lab, lo, hi, seed=s)
        ts.append(nb['crossing'](P, T_lab, T_grid)[0]); accs.append(a)
    allT += ts
    print(f"ends <= {lo}, >= {hi}: validation {np.round(accs, 3)}, crossings {np.round(ts, 2)}")
print(f"all ends and seeds: {min(allT):.2f}-{max(allT):.2f}")
