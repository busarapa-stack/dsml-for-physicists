"""E10.4  Test learning by extremes on a system with a known answer: the 16x16 Ising model (T_c = 2.269).

train_phase_classifier and crossing from code:polymer-nn, applied to the 256 raw spins of code:ising-configs
(data/ising.npz from 05_ising_configs.py, generated here if missing, about one minute).
Answer in the text: ends (1.8, 2.8) -> 2.39-2.41; ends (2.0, 2.6) -> 2.35-2.36; <|m|> ~ 0.58 at T = 2.4;
ends (1.6, 3.2) -> 1.65 with mean P = 0 at T = 1.7, because every configuration at T = 1.6 has m < 0 and
every one at T = 1.7 has m > 0; with spin-flip augmentation (s -> -s) the crossing returns to 2.43-2.45.
About 2 minutes.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from _shared import listing

os.chdir(HERE)
if not Path('data/ising.npz').exists():
    subprocess.run([sys.executable, '05_ising_configs.py'], check=True)
d = np.load('data/ising.npz')
spins, T_spin = d['spins'], d['T_spin']
T_ising = np.unique(T_spin)

nb = {'np': np, 'torch': torch, 'nn': nn}
listing(nb, '04_polymer_classifier.py', 'polymer-nn', 'model, X_all, P, acc =')   # definitions only
train_phase_classifier, crossing = nb['train_phase_classifier'], nb['crossing']

m = spins.mean(axis=1)
print(f"<|m|> at T = 2.4: {np.abs(m[T_spin == 2.4]).mean():.2f}")
print("fraction of configurations with m > 0:",
      ' '.join(f"{t:.1f}:{np.mean(m[T_spin == t] > 0):.2f}" for t in T_ising))

for lo, hi in [(1.8, 2.8), (2.0, 2.6), (1.6, 3.2)]:
    res = []
    for seed in range(3):
        _, _, P, acc = train_phase_classifier(spins, T_spin, lo, hi, seed=seed)
        ts, Pm = crossing(P, T_spin, T_ising)
        res.append(ts)
        if seed == 0:
            first = Pm
    print(f"ends <= {lo}, >= {hi}: crossings {np.round(res, 3)}  (T_c = 2.269)")
    if lo == 1.6:
        print("   mean P by T (seed 0):", ' '.join(f"{t:.1f}:{p:.2f}" for t, p in zip(T_ising, first)))

# symmetry s -> -s: add every configuration with all spins flipped
S2, T2 = np.concatenate([spins, -spins]), np.concatenate([T_spin, T_spin])
res = [crossing(train_phase_classifier(S2, T2, 1.6, 3.2, seed=seed)[2], T2, T_ising)[0] for seed in range(3)]
print(f"ends <= 1.6, >= 3.2 with spin-flip augmentation: crossings {np.round(res, 3)}")
