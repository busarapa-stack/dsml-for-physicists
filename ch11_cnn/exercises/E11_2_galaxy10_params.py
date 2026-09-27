"""E11.2  Parameter counts: GalaxyCNN for 3-band, 10-class Galaxy10 images vs the draft's flatten-and-dense design.

Answer in the text: first layer 160 -> 448, head 650, other layers 4,640 / 18,496 / 36,928, total 61,162;
draft (69x69 input, conv 32/64/128, flatten 128x8x8 -> 256 -> 10): conv 896 / 18,496 / 73,856, dense 2,097,408 and 2,570,
total 2,193,226 (> 95 % in the first dense layer); global average pooling uses ~36 times fewer parameters.
"""
import sys
from pathlib import Path

import torch.nn as nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from _shared import listing

ns = listing({'__name__': 'x'}, '01_cnn_architecture.py', 'cnn-arch')
m = ns['GalaxyCNN'](n_classes=10)
m.features[0] = nn.Conv2d(3, 16, 3, padding=1)
per = [sum(p.numel() for p in l.parameters()) for l in m.features if isinstance(l, nn.Conv2d)]
head = sum(p.numel() for p in m.head.parameters())
total = sum(p.numel() for p in m.parameters())
print(f"3-band GalaxyCNN: conv layers {per}, head {head}, total {total:,}")

draft = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                      nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                      nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                      nn.Flatten(), nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, 10))
import torch
print("draft output shape for a 69x69 image before flattening:", tuple(draft[:9](torch.zeros(1, 3, 69, 69)).shape))
per_d = [sum(p.numel() for p in l.parameters()) for l in draft if isinstance(l, (nn.Conv2d, nn.Linear))]
tot_d = sum(per_d)
print(f"draft: layers {per_d}, total {tot_d:,}; first dense layer {100 * per_d[3] / tot_d:.1f} %; ratio {tot_d / total:.1f}")
