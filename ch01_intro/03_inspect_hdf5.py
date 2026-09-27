"""Ch.1  Walk an HDF5 file (groups, datasets, attributes) and check the numbers
quoted in Section 1.2.2 for the LIGO strain file.

Book references
    sssec:ligo, code:ligo-hdf5, misconception box "strain 1e-21 is a distance"

Checks
    sampling rate comes from the Strain attribute Xspacing (no SampleRate in meta/)
    32 s x 4096 Hz = 131,072 samples
    dL = h * L with h ~ 1e-21 and L = 4 km  ->  4e-18 m, a few hundred times
    smaller than a proton (~1e-15 m)
"""
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "outputs"
OUT.mkdir(exist_ok=True)
path = next(DATA.glob("H-H1_*.hdf5"))


def show(name, obj):
    depth = name.count("/")
    pad = "  " * depth
    if isinstance(obj, h5py.Group):
        print(f"{pad}{name.split('/')[-1]}/")
    else:
        val = obj[()] if obj.shape == () else f"array shape={obj.shape} dtype={obj.dtype}"
        if isinstance(val, bytes):
            val = val.decode()
        print(f"{pad}{name.split('/')[-1]:14s} {val}")
    for k, v in obj.attrs.items():
        print(f"{pad}   @{k} = {v}")


with h5py.File(path, "r") as h:
    print(path.name)
    h.visititems(show)
    # the sampling rate is NOT in meta/: read it from the Strain attribute
    dt = h["strain/Strain"].attrs["Xspacing"]
    fs = int(round(1 / dt))
    duration = int(h["meta/Duration"][()])
    strain = h["strain/Strain"][:]
    dq = h["quality/simple/DQmask"][()]

n = len(strain)
print(f"\nsamples = {n:,}  (expected {duration} s x {fs} Hz = {duration * fs:,})")
assert n == duration * fs
print(f"sampling interval from attribute Xspacing = {dt} s -> {fs} Hz")
print(f"DQmask: {len(dq)} values (one per second), all == 127 (7 flags passed): {bool((dq == 127).all())}")

# The misconception box: strain is a ratio; only dL = h L is a length
h_peak, L = 1e-21, 4e3
dL = h_peak * L
proton = 1e-15
print(f"dL = h L = {h_peak:.0e} x {L:.0f} m = {dL:.1e} m")
print(f"proton diameter / dL = {proton / dL:.0f}  -> 'several hundred times smaller'")

t = np.arange(n) / fs
fig, ax = plt.subplots(2, 1, figsize=(7, 5))
ax[0].plot(t, strain, lw=0.3, color="k")
ax[0].set_xlabel("time since GPS start [s]"); ax[0].set_ylabel("strain h")
ax[0].set_title("SYNTHETIC strain, 32 s")
m = (t > 16.0) & (t < 16.5)
ax[1].plot(t[m], strain[m], lw=0.6, color="k")
ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("strain h")
ax[1].set_title("zoom: toy chirp buried in noise")
fig.tight_layout()
fig.savefig(OUT / "ch01_strain.png", dpi=150)
print("saved outputs/ch01_strain.png")
