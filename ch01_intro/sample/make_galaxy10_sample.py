"""Ch.1  Build the small Galaxy10 DECaLS sample used by notebooks/ch01_intro_classroom.ipynb.

The full file Galaxy10_DECals.h5 (about 2.7 GB, https://zenodo.org/records/10845026)
is too large to ship with the companion code and is slow to read over the network
in Colab. This script copies a small, UNMODIFIED subset into
ch01_intro/sample/galaxy10_decals_sample.h5 (about 7 MB):

    images, ans, ra, dec, redshift, pxscale   first 5 images of each of the 10 classes (50 rows)
    index_in_full                             row numbers of those 50 images in the full file
    ans_full                                  labels of all 17,736 images (for class counts)

Usage (from the code/ folder, with the full file present):
    python ch01_intro/sample/make_galaxy10_sample.py [path/to/Galaxy10_DECals.h5]

Data: Galaxy10 DECaLS, H. W. Leung & J. Bovy (CC BY 4.0); labels from Galaxy Zoo
DECaLS (Walmsley et al. 2022, MNRAS 509, 3966); images from the DESI Legacy
Imaging Surveys.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

N_PER_CLASS = 5
HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / "Galaxy10_DECals.h5"
OUT = HERE / "galaxy10_decals_sample.h5"

with h5py.File(SRC, "r") as f:
    labels = f["ans"][()]
    idx = np.sort(np.concatenate([np.where(labels == c)[0][:N_PER_CLASS] for c in range(10)]))
    with h5py.File(OUT, "w") as g:
        g.create_dataset("images", data=f["images"][idx], compression="gzip",
                         compression_opts=9, chunks=(1, 256, 256, 3))
        for key in ("ans", "ra", "dec", "redshift", "pxscale"):
            g.create_dataset(key, data=f[key][idx])
        g.create_dataset("index_in_full", data=idx.astype("int32"))
        g.create_dataset("ans_full", data=labels, compression="gzip")
        g.attrs["description"] = ("Subset of Galaxy10 DECaLS: first 5 images of each of the 10 classes "
                                  "(50 images, unmodified 256x256x3 uint8), plus the labels of all "
                                  "17,736 images for class counts.")
        g.attrs["source"] = ("Galaxy10_DECals.h5, Leung & Bovy, https://zenodo.org/records/10845026 ; "
                             "labels from Galaxy Zoo DECaLS (Walmsley et al. 2022, MNRAS 509, 3966); "
                             "images from the DESI Legacy Imaging Surveys")
        g.attrs["license"] = "CC BY 4.0 (as stated for the source dataset); attribution required"
        g.attrs["made_by"] = "ch01_intro/sample/make_galaxy10_sample.py"

print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB), {len(idx)} images")
print("class counts:", np.bincount(labels, minlength=10).tolist())
