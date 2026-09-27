"""E11.6  Where does the clean CNN look? Share of the Grad-CAM map inside radius 20 px, per class (true class as target).

Uses grad_cam of code:gradcam and models['CNN'] (data/CNN.pt from 03_train_evaluate.py).
Answer in the text: the circle is ~31 % of the image; share inside ~86 % (spiral), 68 % (edge-on), 59 % (barred),
but only ~15 % (smooth round) and 22 % (cigar) - less than the area share.
About 20 seconds.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from _shared import book_namespace, section

os.chdir(HERE)
ns = book_namespace(trained=True)
exec(section(HERE / '06_gradcam_shortcut.py', '\n# --- code:gradcam', '\ndef add_star'), ns)    # grad_cam only
grad_cam, model, X_test, y_test, CLASSES = ns['grad_cam'], ns['models']['CNN'], ns['X_test'], ns['y_test'], ns['CLASSES']

yy, xx = np.mgrid[:64, :64]
disc = (yy - 31.5)**2 + (xx - 31.5)**2 < 20**2
print(f"area share of the r < 20 disc: {disc.mean():.3f}")
for c in range(5):
    imgs = X_test[y_test == c][:50]
    share = [grad_cam(model, torch.tensor(im[None, None]), c)[disc].sum() for im in imgs]
    print(f"{CLASSES[c]:14s}: share of Grad-CAM inside {np.mean(share):.2f}")
