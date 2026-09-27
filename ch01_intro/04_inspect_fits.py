"""Ch.1  Read FITS files: header keywords (metadata) and data (image or table).

Book references
    sssec:sdss, code:fits-header, exercise ex:matching items (b) and (e)

The same file format holds two modalities here:
    sdss_like_image_r.fits   -> image    (2D array in the primary HDU)
    sdss_like_spectrum.fits  -> spectrum (binary table in extension 1)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "outputs"
OUT.mkdir(exist_ok=True)

# ---- image -----------------------------------------------------------------
with fits.open(DATA / "sdss_like_image_r.fits") as hdul:
    hdul.info()
    hdr, img = hdul[0].header, hdul[0].data

print("\nheader (compare with code:fits-header):")
for key in ("SIMPLE", "NAXIS", "NAXIS1", "NAXIS2", "RA", "DEC", "FILTER", "EXPTIME"):
    print(f"  {key:8s}= {hdr[key]!r:>20}  / {hdr.comments[key]}")

print(f"\nimage shape {img.shape}, dtype {img.dtype}")
print(f"pixel values: min {img.min():.1f}, median {np.median(img):.2f}, max {img.max():.1f}")
print(f"counts per second at the brightest pixel: {img.max() / hdr['EXPTIME']:.2f}"
      "   <- needs EXPTIME from the header")

# ---- spectrum --------------------------------------------------------------
with fits.open(DATA / "sdss_like_spectrum.fits") as hdul:
    print()
    hdul.info()
    tab = hdul["COADD"].data
lam = 10 ** tab["loglam"]
flux = tab["flux"]
print(f"\nspectrum: {len(lam)} pixels, {lam.min():.0f}-{lam.max():.0f} Angstrom")
print(f"wavelength step grows with lambda: {lam[1]-lam[0]:.3f} A at blue end, "
      f"{lam[-1]-lam[-2]:.3f} A at red end (constant step in log10)")
i_ha = np.argmin(np.abs(lam - 6563))
win = (lam > 6500) & (lam < 6620)
print(f"flux at H-alpha 6563 A is {flux[i_ha] / np.median(flux[win]):.2f} of the local median"
      " -> absorption line")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(np.arcsinh(img / 5), origin="lower", cmap="gray")
ax[0].set_title(f"SYNTHETIC r-band image  {img.shape[1]}x{img.shape[0]} px")
ax[1].plot(lam, flux, lw=0.5, color="k")
for line in (4340, 4861, 5892, 6563):
    ax[1].axvline(line, ls=":", color="gray")
ax[1].set_xlabel(r"wavelength [$\AA$]"); ax[1].set_ylabel("flux [arb.]")
ax[1].set_title("SYNTHETIC stellar spectrum")
fig.tight_layout()
fig.savefig(OUT / "ch01_fits_image_spectrum.png", dpi=150)
print("saved outputs/ch01_fits_image_spectrum.png")
