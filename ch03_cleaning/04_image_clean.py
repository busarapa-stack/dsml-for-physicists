"""Ch.3  The three groups of image problems in ssec:image: hot pixels (and why a 5-sigma
cut is not enough), noise smoothing, and uneven illumination corrected with a flat field.

Book references: ssec:image, code:image-clean, tab:cleaning-techniques (row "image")
Images are SYNTHETIC (4 x 256 x 256 x 3); image 0, band 0 has two hot pixels.
"""
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

os.chdir(Path(__file__).resolve().parent)
images = np.load('data/raw/galaxy_images.npy')
print("images:", images.shape, images.dtype)

# --- code:image-clean -------------------------------------------------------
from scipy.ndimage import gaussian_filter

img = images[0, :, :, 0].astype(float)      # first image, first band
threshold = img.mean() + 5 * img.std()
hot_mask = img > threshold

img_smooth = gaussian_filter(img, sigma=1.0)

# --- what the 5-sigma cut actually found -------------------------------------
ys, xs = np.nonzero(hot_mask)
print(f"threshold = {threshold:.2f}; pixels above it: {hot_mask.sum()}")
near_centre = np.hypot(xs - 128, ys - 128) < 10
print(f"  of these, {near_centre.sum()} are in the galaxy core (real light), "
      f"{(~near_centre).sum()} are elsewhere")

# better test: bright AND very different from the median of its neighbours
local = median_filter(img, size=3)
noise = 1.4826 * np.median(np.abs(img - np.median(img)))       # robust sigma
hot = (img - local) > 10 * noise
print(f"neighbour test (img - local median > 10 sigma): {hot.sum()} pixels at "
      f"{list(zip(*np.nonzero(hot)))}  <- the two injected hot pixels")

fixed = np.where(hot, local, img)
print(f"after replacing them by the local median: max outside the core = "
      f"{fixed[np.hypot(*np.mgrid[0:256, 0:256] - 128) > 20].max():.2f}")
print(f"Gaussian smoothing (sigma = 1 px): noise std {np.std(img[:30, :30]):.3f} -> "
      f"{np.std(img_smooth[:30, :30]):.3f}, peak {img[118:138, 118:138].max():.1f} -> "
      f"{img_smooth[118:138, 118:138].max():.1f} (the core is blurred too)")

# --- third group in the text: vignetting, corrected by dividing by a flat field ----
yy, xx = np.mgrid[0:256, 0:256]
rr = np.hypot(xx - 127.5, yy - 127.5) / 181.0            # 0 at centre, 1 at the corners
vignette = 1 - 0.4 * rr**2                                # corners 40 % darker
observed = fixed * vignette
flat = vignette * (1 + np.random.default_rng(4).normal(0, 0.002, vignette.shape))  # flat-field frame
corrected = observed / flat
corner, centre = (slice(0, 30), slice(0, 30)), (slice(113, 143), slice(113, 143))
print(f"vignetting: corner/centre sky ratio {vignette[corner].mean() / vignette[centre].mean():.2f};"
      f" after dividing by the flat field the image equals the original to within "
      f"{np.median(np.abs(corrected - fixed)[fixed > 2] / fixed[fixed > 2]):.1%} (median)")
