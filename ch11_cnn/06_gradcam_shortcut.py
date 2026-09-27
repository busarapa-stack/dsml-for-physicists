"""Ch.11  A deliberate shortcut (bright spot on 90 % of barred spirals) and how Grad-CAM exposes it.

Book references: ssec:gradcam, code:gradcam (line for line)
Continues from code:cnn-train (fit, predict, models['CNN'] from data/CNN.pt).
Numbers quoted in the text: validation (with the artefact) 0.968; clean test 0.949, barred recall 0.91
(clean CNN 0.978 = 391/400); spot on every test image: accuracy 0.20, all called barred;
Grad-CAM share in the 16x16 corner (6 % of the area): shortcut 0.53, clean CNN 0.05.
About 3 minutes.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import book_namespace

os.chdir(HERE)
Path('outputs').mkdir(exist_ok=True)
globals().update(book_namespace(trained=True))

# --- code:gradcam ------------------------------------------------------------
import torch.nn.functional as F

def grad_cam(model, x, target):
    """Grad-CAM map (64 x 64, unit sum) of class `target` for one image x (1,1,64,64)."""
    model.eval()
    A = model.features(x)                          # last feature maps (1, 64, 8, 8)
    A.retain_grad()
    score = model.head(A.mean(dim=(2, 3)))[0, target]
    model.zero_grad()
    score.backward()
    w = A.grad.mean(dim=(2, 3), keepdim=True)      # importance of each feature map
    cam = F.relu((w * A).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
    cam = cam[0, 0].detach().numpy()
    return cam / (cam.sum() + 1e-12)

def add_star(X):                                   # a 3x3 bright spot near one corner
    X = X.copy(); X[:, 4:7, 4:7] += 1.5
    return X

rng = np.random.default_rng(7)                     # 90% of barred training images get it
X_tr_s = X_train.copy(); m = (y_train == 4) & (rng.random(len(y_train)) < 0.9)
X_tr_s[m] = add_star(X_tr_s[m])
X_va_s = X_val.copy(); m = (y_val == 4) & (rng.random(len(y_val)) < 0.9)
X_va_s[m] = add_star(X_va_s[m])
torch.manual_seed(0)
shortcut = GalaxyCNN()
val_acc = fit(shortcut, X_tr_s, y_train, X_va_s, y_val)
print("validation accuracy (with the artefact):", round(val_acc[-1], 3))
pred = predict(shortcut, X_test).argmax(1)
print("clean test:", np.mean(pred == y_test).round(3),
      " barred recall:", np.mean(pred[y_test == 4] == 4).round(3))
pred = predict(shortcut, add_star(X_test)).argmax(1)
print("spot on every test image: accuracy", np.mean(pred == y_test).round(3),
      " fraction called barred", np.mean(pred == 4).round(3))

yy, xx = np.mgrid[:64, :64]
corner = (yy < 16) & (xx < 16)                     # 6% of the image area
for name, model in [('clean CNN', models['CNN']), ('shortcut CNN', shortcut)]:
    frac = [grad_cam(model, torch.tensor(img[None, None]), 4)[corner].sum()
            for img in add_star(X_test[y_test == 4][:100])]
    print(f"{name}: share of Grad-CAM in the corner = {np.mean(frac):.3f}")
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    pred0 = predict(models['CNN'], X_test).argmax(1)
    print(f"clean CNN: barred recall {np.mean(pred0[y_test == 4] == 4):.3f}; corner area {corner.mean():.3f}")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    img = add_star(X_test[y_test == 4][:1])[0]
    fig, ax = plt.subplots(1, 3, figsize=(8, 2.8))
    ax[0].imshow(img, cmap='gray', origin='upper'); ax[0].set_title('barred spiral + spot', fontsize=9)
    for a, (name, model) in zip(ax[1:], [('clean CNN', models['CNN']), ('shortcut CNN', shortcut)]):
        a.imshow(img, cmap='gray', origin='upper')
        a.imshow(grad_cam(model, torch.tensor(img[None, None]), 4), cmap='inferno', alpha=0.55, origin='upper')
        a.set_title(f'Grad-CAM, {name}', fontsize=9)
    for a in ax:
        a.axis('off')
    fig.tight_layout(); fig.savefig('outputs/ch11_gradcam.png', dpi=100)
    print("saved outputs/ch11_gradcam.png")
