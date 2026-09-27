"""E10.6  How much of the network uncertainty is a choice? Dropout rate and training ends.

1. Retrain the classifier of code:polymer-nn with dropout 0.05 and 0.5 (instead of 0.2) and repeat MC dropout
   (code:mc-dropout): the dropout SD changes with the rate while the crossing stays put.
2. Train the 5-network ensemble with ends (T <= 0.8, T >= 3.2) and compare the crossings with the
   original ensemble (ends 1.0 / 3.0: 1.33-1.40).
Answer in the text: maximum dropout SD ~0.07 (rate 0.2), ~0.04 (0.05), ~0.14 (0.5); crossing still 1.33;
ensemble with ends 0.8 / 3.2: crossings 1.11-1.25, no overlap with 1.33-1.40, wider spread.
About 1.5 minutes.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from _shared import chain_data, section

os.chdir(HERE)
nb = chain_data({'np': np, 'torch': torch, 'nn': nn})
T_lab, T_grid, dist = nb['T_lab'], nb['T_grid'], nb['dist']
defs = section(HERE / '04_polymer_classifier.py', '\n# --- code:polymer-nn', 'model, X_all, P, acc =')
exec(section(HERE / '07_mc_dropout_ensemble.py', '\n# --- code:mc-dropout', 'torch.manual_seed(1)'), nb)   # mc_dropout_predict

for rate in (0.05, 0.2, 0.5):
    exec(defs.replace('nn.Dropout(0.2)', f'nn.Dropout({rate})'), nb)
    model, X_all, P, acc = nb['train_phase_classifier'](dist, T_lab, T_low=1.0, T_high=3.0)
    torch.manual_seed(1)
    p_mc, s_mc = nb['mc_dropout_predict'](model, X_all)
    sd = np.array([s_mc[T_lab == t].mean() for t in T_grid])
    print(f"dropout {rate:4.2f}: val. acc {acc:.3f}, crossing {nb['crossing'](P, T_lab, T_grid)[0]:.2f}, "
          f"max MC-dropout SD {sd.max():.3f} at T = {T_grid[sd.argmax()]:.1f}")

exec(defs, nb)                                                 # back to dropout 0.2
for lo, hi in [(1.0, 3.0), (0.8, 3.2)]:
    ts = [nb['crossing'](nb['train_phase_classifier'](dist, T_lab, lo, hi, seed=s)[2], T_lab, T_grid)[0] for s in range(5)]
    print(f"ensemble with ends <= {lo}, >= {hi}: crossings {np.round(ts, 2)}, range {np.ptp(ts):.2f}, SD {np.std(ts, ddof=1):.3f}")
