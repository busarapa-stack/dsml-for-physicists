"""Ch.11  Toy galaxy images of five classes from Sersic profiles, discs, logarithmic arms and bars.

Book references: ssec:galaxy-sim, code:galaxy-gen (line for line)
Writes data/galaxies.npz (train 5,000, validation 1,250, test 2,000 images), reused by every later script,
and outputs/ch11_galaxy_examples.png. "A few seconds" to generate.
"""
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
Path('data').mkdir(exist_ok=True)
Path('outputs').mkdir(exist_ok=True)
_t0 = time.time()

# --- code:galaxy-gen ---------------------------------------------------------
import numpy as np
from scipy.ndimage import gaussian_filter
CLASSES = ['smooth round', 'smooth cigar', 'edge-on disk', 'spiral', 'barred spiral']

def make_galaxy(cls, rng, size=64):
    """One noisy 64x64 image of a toy galaxy of class cls (0-4)."""
    y, x = np.mgrid[:size, :size] - (size - 1) / 2
    x0, y0 = rng.normal(0, 1.5, 2)                     # small centring error
    x, y = x - x0, y - y0
    pa = rng.uniform(0, np.pi)                          # position angle
    xr = x * np.cos(pa) + y * np.sin(pa)
    yr = -x * np.sin(pa) + y * np.cos(pa)
    def sersic(re, n, q):
        r = np.sqrt(xr**2 + (yr / q)**2)
        b = 2 * n - 1 / 3
        return np.exp(-b * ((r / re)**(1 / n) - 1))
    if cls == 0:
        img = sersic(rng.uniform(6, 10), 4, rng.uniform(0.8, 1.0))
    elif cls == 1:
        img = sersic(rng.uniform(6, 10), rng.uniform(1, 4), rng.uniform(0.2, 0.4))
    elif cls == 2:
        img = sersic(rng.uniform(10, 15), 1, rng.uniform(0.08, 0.15)) \
              + 0.3 * sersic(2.5, 2, 0.6)
    else:
        q = rng.uniform(0.45, 1.0)                      # cos(inclination)
        re = rng.uniform(10, 15)
        r = np.sqrt(xr**2 + (yr / q)**2)
        theta = np.arctan2(yr / q, xr)
        pitch = rng.uniform(0.25, 0.45)                 # arm winding
        arms = 0.5 * (1 + np.cos(2 * (theta - np.log(r + 1e-3) / np.tan(pitch))))
        arms = arms * (1 - np.exp(-(r / (0.4 * re))**2))  # arms start outside the centre
        disk = np.exp(-r / (re / 1.68)) * (0.4 + 1.2 * arms**2)
        img = disk + 1.5 * sersic(2.0, 2, 0.7 + 0.3 * q) / np.exp(11 / 3)   # bulge, rounder than disk
        if cls == 4:                                     # bar along the disk
            bar_len = rng.uniform(0.4, 0.6) * re
            img += 1.2 * np.exp(-(xr / bar_len)**4 - (yr / (0.3 * bar_len))**2)
    img = np.arcsinh(img / 0.3)                        # astronomical asinh stretch
    img = img / img.max() * rng.uniform(0.6, 1.0)
    img = gaussian_filter(img, rng.uniform(0.8, 1.5))   # seeing (PSF)
    img = img + rng.normal(0, 0.05, img.shape)          # sky + read noise
    return img.astype(np.float32)

def make_galaxy_set(n_per_class, seed=0):
    rng = np.random.default_rng(seed)
    X = np.array([make_galaxy(c, rng) for c in range(5) for _ in range(n_per_class)])
    y = np.repeat(np.arange(5), n_per_class)
    p = rng.permutation(len(y))
    return X[p], y[p]

X_train, y_train = make_galaxy_set(1000, seed=1)     # 5000 images
X_val, y_val = make_galaxy_set(250, seed=2)          # 1250 images
X_test, y_test = make_galaxy_set(400, seed=3)        # 2000 images
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    print(f"generated {len(X_train) + len(X_val) + len(X_test)} images in {time.time() - _t0:.1f} s; "
          f"shapes {X_train.shape}, {X_val.shape}, {X_test.shape}; class counts {np.bincount(y_train)}")
    np.savez_compressed('data/galaxies.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test)
    print("saved data/galaxies.npz")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 5, figsize=(10, 6.3))
    for c in range(5):
        for r, i in enumerate(np.where(y_test == c)[0][:3]):
            ax[r, c].imshow(X_test[i], cmap='gray', origin='lower'); ax[r, c].axis('off')
        ax[0, c].set_title(CLASSES[c], fontsize=10)
    fig.tight_layout(); fig.savefig('outputs/ch11_galaxy_examples.png', dpi=90)
    print("saved outputs/ch11_galaxy_examples.png")
