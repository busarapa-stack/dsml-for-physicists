"""Ch.11 (optional)  Read Galaxy10 DECaLS and downsample to 64x64 (code:galaxy10, line for line).

Needs the real file from Zenodo (about 2.7 GB, CC BY 4.0, Leung et al.): put Galaxy10_DECals.h5 in this folder, in ch11_cnn/data/, ch11_cnn/ or code/,
or the de-duplicated version renamed to that name (see ssec:galaxy10-data and E11.4). Also needs h5py.
Reads 1,000 images at a time (the whole file as float32 would need about 14 GB). Checks the statements of ssec:galaxy10-data: 17,736 images of 256 x 256 x 3, ten classes from 334 (cigar-shaped smooth)
to 2,645 (round smooth). The de-duplicated file has fewer images; the counts printed then differ.
"""
import os
from pathlib import Path

import numpy as np
import torch

_here = Path(__file__).resolve().parent                  # look in optional/, ch11_cnn/data/, ch11_cnn/, code/
_where = [d for d in (_here, _here.parent / 'data', _here.parent, _here.parent.parent)
          if (d / 'Galaxy10_DECals.h5').exists()]
if not _where:
    raise SystemExit("Galaxy10_DECals.h5 not found in optional/, ch11_cnn/data/, ch11_cnn/ or code/ - download it from Zenodo first (about 2.7 GB)")
os.chdir(_where[0])

# --- code:galaxy10 -----------------------------------------------------------
import h5py
import torch.nn.functional as F

# 1,000 images at a time: all 17,736 as float32 at once would need about 14 GB
with h5py.File('Galaxy10_DECals.h5', 'r') as f:   # about 2.7 GB, from Zenodo
    labels = f['ans'][:].astype(np.int64)          # (17736,) classes 0-9
    x = torch.empty(len(labels), 3, 64, 64)
    for i in range(0, len(labels), 1000):          # images: (17736, 256, 256, 3) uint8, g r z
        batch = torch.tensor(f['images'][i:i + 1000]).permute(0, 3, 1, 2).float() / 255.0
        x[i:i + 1000] = F.avg_pool2d(batch, kernel_size=4)   # downsample to 64 x 64
print(x.shape, np.bincount(labels))

from sklearn.model_selection import train_test_split
idx_tr, idx_te = train_test_split(np.arange(len(labels)), test_size=0.2,
                                  stratify=labels, random_state=42)
# GalaxyCNN needs nn.Conv2d(3, 16, ...) as its first layer and n_classes=10
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    names = ['disturbed', 'merging', 'round smooth', 'in-between round smooth', 'cigar-shaped smooth',
             'barred spiral', 'unbarred tight spiral', 'unbarred loose spiral', 'edge-on without bulge',
             'edge-on with bulge']
    counts = np.bincount(labels, minlength=10)
    for n, c in zip(names, counts):
        print(f"  {n:25s} {c:5d}")
    print(f"images {len(labels):,}, smallest class {names[counts.argmin()]} ({counts.min()}), "
          f"largest {names[counts.argmax()]} ({counts.max()}); train/test {len(idx_tr)}/{len(idx_te)}")
