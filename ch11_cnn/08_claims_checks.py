"""Ch.11  Second-pass checks: statements without their own listing, and how much single-run numbers move.

  1  all eight main listings run in book order as one notebook (models kept in memory, not reloaded)
  2  output size floor((n + 2p - k)/s) + 1 and parameters C_out (C_in k^2 + 1)             (def:conv)
  3  receptive fields of the four conv outputs 3, 8, 18, 38 (4, 10, 22 after pooling)     (ssec:feature-map)
  4  shift the input by 5 px: a conv layer's map shifts by 5 px, values unchanged (away from edges)  (ssec:cnn-intuition)
  5  MLP: almost all parameters in the first layer; head 325                               (ssec:pytorch-conv)
  6  shifted vs centred CNN accuracy differ within the binomial error of 2,000 test images (ssec:cnn-results)
  7  best validation minus test accuracy, 100 images per class: "five to eight points"     (ssec:augmentation)
  8  Grad-CAM equals the class activation map for GAP + one linear layer; map is 8 x 8    (ssec:gradcam)
  9  ensemble accuracy above almost every member                                           (ssec:cnn-ensemble)
 10  run-to-run spread: MLP and CNN trained with seeds 0-4 (the CNNs are the ensemble members of code:ens-ood):
     test and shifted accuracy, and the largest off-diagonal cells of each confusion matrix
About 30 minutes (item 1 dominates; the ensemble alone is about 12 minutes).
"""
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import section

os.chdir(HERE)

# 1 ------------------------------------------------------------------------------------------------
t0 = time.time()
nb = {'__name__': 'notebook'}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    for script, first in [('01_cnn_architecture.py', 'cnn-arch'), ('02_galaxy_images.py', 'galaxy-gen'),
                          ('03_train_evaluate.py', 'cnn-train'), ('04_small_augmentation.py', 'small-aug'),
                          ('05_transfer_rice.py', 'transfer'), ('06_gradcam_shortcut.py', 'gradcam'),
                          ('07_ensemble_ood.py', 'ens-ood')]:
        exec(section(HERE / script, f'\n# --- code:{first}', '\n# --- end of listings'), nb)
out = buf.getvalue()
print(f"1  eight listings as one notebook: {time.time() - t0:.0f} s")
for key in ('CNN: test', 'MLP: test', 'augmentation=', 'fine-tune', 'shortcut CNN', 'ensemble accuracy', 'stars', 'mergers'):
    for l in out.splitlines():
        if l.startswith(key):
            print('   ', l)
globals().update({k: nb[k] for k in ('GalaxyCNN', 'MLP', 'X_test', 'y_test', 'X_shift', 'X_train', 'y_train',
                                      'X_val', 'y_val', 'fit', 'predict', 'models', 'members', 'grad_cam')})

# 2 ------------------------------------------------------------------------------------------------
for n, k, p, s in [(64, 3, 1, 1), (64, 5, 0, 1), (64, 3, 1, 2), (69, 3, 1, 1)]:
    conv = nn.Conv2d(3, 16, k, stride=s, padding=p)
    out_w = conv(torch.zeros(1, 3, n, n)).shape[-1]
    npar = sum(q.numel() for q in conv.parameters())
    print(f"2  n={n} k={k} p={p} s={s}: torch {out_w}, formula {(n + 2 * p - k) // s + 1}; params {npar} = 16(3*{k * k}+1) = {16 * (3 * k * k + 1)}")

# 3 ------------------------------------------------------------------------------------------------
m = GalaxyCNN()
f = nn.Sequential(*[nn.AvgPool2d(2) if isinstance(l, nn.MaxPool2d) else l for l in m.features])   # same geometry
for q in f.parameters():
    nn.init.constant_(q, 0.01) if q.dim() == 1 else nn.init.uniform_(q, 0.5, 1.0)
rf = []
for i, l in enumerate(f):
    if isinstance(l, (nn.Conv2d, nn.AvgPool2d)):
        x = torch.ones(1, 1, 64, 64, requires_grad=True)
        y = f[:i + 2 if isinstance(l, nn.Conv2d) else i + 1](x)
        c = y.shape[-1] // 2
        y[0, 0, c, c].backward()
        rows = (x.grad[0, 0] != 0).any(1).nonzero()
        rf.append(('conv' if isinstance(l, nn.Conv2d) else 'pool', int(rows.max() - rows.min() + 1)))
print("3  receptive fields:", rf)

# 4 ------------------------------------------------------------------------------------------------
conv1 = models['CNN'].features[0]
img = torch.tensor(X_test[:1, None])
with torch.no_grad():
    a = conv1(img); b = conv1(torch.roll(img, 5, dims=3))
print(f"4  shift by 5 px: max |conv(shift x) - shift conv(x)| inside the image = "
      f"{(b[..., 8:-8] - torch.roll(a, 5, dims=3)[..., 8:-8]).abs().max().item():.1e}")

# 5 ------------------------------------------------------------------------------------------------
mlp = MLP()
first = sum(q.numel() for q in mlp.net[1].parameters())
print(f"5  MLP first layer {first:,} of {sum(q.numel() for q in mlp.parameters()):,} ({100 * first / sum(q.numel() for q in mlp.parameters()):.1f} %); "
      f"CNN head {sum(q.numel() for q in GalaxyCNN().head.parameters())}")

# 6 ------------------------------------------------------------------------------------------------
acc = np.mean(predict(models['CNN'], X_test).argmax(1) == y_test)
acc_s = np.mean(predict(models['CNN'], X_shift).argmax(1) == y_test)
print(f"6  CNN centred {acc:.4f}, shifted {acc_s:.4f}: difference {acc_s - acc:+.4f}, binomial SE of one accuracy "
      f"{np.sqrt(acc * (1 - acc) / 2000):.4f}")

# 7 ------------------------------------------------------------------------------------------------
print(f"7  best validation - test: {100 * (0.814 - 0.737):.1f} and {100 * (0.829 - 0.771):.1f} points (numbers of code:small-aug)")

# 8 ------------------------------------------------------------------------------------------------
cnn = models['CNN']; x1 = torch.tensor(X_test[y_test == 4][:1, None])
with torch.no_grad():
    A = cnn.features(x1)
    cam = torch.relu((cnn.head.weight[4][None, :, None, None] * A).sum(1, keepdim=True))
    cam = nn.functional.interpolate(cam, size=(64, 64), mode='bilinear', align_corners=False)[0, 0].numpy()
cam = cam / cam.sum()
g = grad_cam(cnn, x1, 4)
print(f"8  last feature maps {tuple(A.shape[-2:])}; max |Grad-CAM - CAM| (both unit sum) = {np.abs(g - cam).max():.1e}")

# 9 and 10 ---------------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
names = ['round', 'cigar', 'edge-on', 'spiral', 'barred']


def top_errors(y, p, k=2):
    C = confusion_matrix(y, p); np.fill_diagonal(C, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-C.ravel())[:k], C.shape))[0]
    return ', '.join(f"{names[i]}->{names[j]} {C[i, j]}" for i, j in idx)


member_acc = [np.mean(predict(mm, X_test).argmax(1) == y_test) for mm in members]
P = np.mean([predict(mm, X_test) for mm in members], axis=0)
print(f"9  ensemble {np.mean(P.argmax(1) == y_test):.3f} above {sum(a < np.mean(P.argmax(1) == y_test) for a in member_acc)} of 5 members")
print("10 CNN seeds 0-4 (the ensemble members):")
for s, mm in enumerate(members):
    p = predict(mm, X_test).argmax(1)
    print(f"   seed {s}: test {np.mean(p == y_test):.3f}, shifted {np.mean(predict(mm, X_shift).argmax(1) == y_test):.3f}; "
          f"largest errors {top_errors(y_test, p)}")
print("   MLP seeds 0-4 (15 epochs, lr 1e-3 as in code:cnn-train):")
mlp_acc = []
for s in range(5):
    torch.manual_seed(s)
    mm = MLP()
    fit(mm, X_train, y_train, X_val, y_val, epochs=15, lr=1e-3, seed=s)
    p = predict(mm, X_test).argmax(1)
    mlp_acc.append(np.mean(p == y_test))
    print(f"   seed {s}: test {mlp_acc[-1]:.3f}, shifted {np.mean(predict(mm, X_shift).argmax(1) == y_test):.3f}; "
          f"largest errors {top_errors(y_test, p)}")
print(f"   mean test accuracy: CNN {np.mean(member_acc):.3f} (range {min(member_acc):.3f}-{max(member_acc):.3f}), "
      f"MLP {np.mean(mlp_acc):.3f} (range {min(mlp_acc):.3f}-{max(mlp_acc):.3f})")
