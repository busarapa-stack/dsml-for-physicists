"""Ch.11  A small CNN and an MLP for 64x64 one-channel images: parameter counts.

Book references: ssec:pytorch-conv, code:cnn-arch (line for line), E11.2
Numbers quoted in the text: MLP 532,997 parameters, GalaxyCNN 60,549 ("almost nine times" fewer);
head 64 x 5 + 5 = 325; first MLP layer 4,096 x 128 = 524,288; a 256 x 256 x 3 input into 128 units: "over 25 million".
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

# --- code:cnn-arch -----------------------------------------------------------
import torch
import torch.nn as nn

class GalaxyCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64 -> 32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())                   # 64 maps, 8x8
        self.head = nn.Linear(64, n_classes)
    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))   # global average pooling

class MLP(nn.Module):                                           # for comparison
    def __init__(self, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(64 * 64, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, n_classes))
    def forward(self, x):
        return self.net(x)

for Model in (MLP, GalaxyCNN):
    print(Model.__name__, sum(p.numel() for p in Model().parameters()), "parameters")
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    n_mlp, n_cnn = (sum(p.numel() for p in M().parameters()) for M in (MLP, GalaxyCNN))
    print(f"ratio MLP / CNN = {n_mlp / n_cnn:.2f}; head {sum(p.numel() for p in GalaxyCNN().head.parameters())}; "
          f"MLP first layer weights {64 * 64 * 128:,}; 256x256x3 into 128 units: {256 * 256 * 3 * 128:,}")
    print("per layer (CNN):", [sum(p.numel() for p in l.parameters()) for l in GalaxyCNN().features if isinstance(l, nn.Conv2d)])
