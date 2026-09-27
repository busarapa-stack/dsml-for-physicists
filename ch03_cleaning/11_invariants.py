"""Ch.3  Rotation-invariant image features, and why '==' is the wrong test for floats.

Book references: ssec:symmetry-feat, tab:invariant, code:invariant,
                 misconception box "invariant values must agree to every decimal"
"""
import numpy as np

# --- code:invariant ---------------------------------------------------------
def total_flux(image):
    return image.sum()

def radial_profile(image, center=None):
    if center is None:
        center = ((image.shape[0] - 1) / 2, (image.shape[1] - 1) / 2)
    y, x = np.indices(image.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    profile = np.bincount(r.ravel(), weights=image.ravel())
    counts  = np.bincount(r.ravel())
    return profile / np.maximum(counts, 1)

img     = np.random.default_rng(2026).normal(size=(64, 64))
img_rot = np.rot90(img)
print(np.isclose(total_flux(img), total_flux(img_rot)))   # True
print(np.allclose(radial_profile(img), radial_profile(img_rot)))   # True

# --- the misconception box --------------------------------------------------
print(f"total_flux(img) == total_flux(img_rot): {total_flux(img) == total_flux(img_rot)}")
print(f"difference: {total_flux(img) - total_flux(img_rot):.1e}  (order of the additions differs)")

# why the default centre is ((N-1)/2, (N-1)/2): rot90 turns the pixel grid about that
# point; for a 64 x 64 image it is (31.5, 31.5), not the pixel (32, 32)
c = (np.array(img.shape) - 1) / 2
prof, prof_rot = radial_profile(img), radial_profile(img_rot)
print(f"radial profile about ({c[0]}, {c[1]}): max |difference| after rotation = "
      f"{np.abs(prof - prof_rot).max():.1e}")
p32, p32r = radial_profile(img, center=(32, 32)), radial_profile(img_rot, center=(32, 32))
print(f"radial profile about the pixel (32, 32): max |difference| = "
      f"{np.abs(p32 - p32r).max():.2f}  <- not invariant: wrong rotation centre")

# first moment is NOT invariant
y, x = np.indices(img.shape)
print(f"first moment sum(x*I): {np.sum(x * img):.2f} -> after rotation {np.sum(x * img_rot):.2f}")

# concentration and asymmetry on a smooth synthetic galaxy
r = np.hypot(x - c[1], y - c[0])
gal = np.exp(-r / 6.0)
def radius_containing(image, frac):
    rr = np.hypot(*(np.indices(image.shape) - c[:, None, None]))
    order = np.argsort(rr.ravel())
    cum = np.cumsum(image.ravel()[order]) / image.sum()
    return rr.ravel()[order][np.searchsorted(cum, frac)]
C = 5 * np.log10(radius_containing(gal, 0.8) / radius_containing(gal, 0.2))
A = np.abs(gal - np.rot90(gal, 2)).sum() / np.abs(gal).sum()
print(f"exponential disk: concentration C = {C:.2f}, asymmetry A = {A:.1e}")
