"""Ch.1  Open the ROOT tree and see why event-level data is 'jagged'.

Book references
    sssec:cerndata, code:root-tree

What this script shows
    1. the branch list of the tree (compare with code:root-tree)
    2. the number of muons differs from event to event
    3. a jagged array cannot be a plain rectangular table without a choice
       (pad, truncate, or explode to one row per muon)
    4. a first physics look: the dimuon invariant mass shows the Z peak
"""
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "outputs"
OUT.mkdir(exist_ok=True)

tree = uproot.open(DATA / "dimuon_sample.root")["Events"]

# 1) structure ---------------------------------------------------------------
print(f"Tree 'Events': {tree.num_entries} events")
for name, br in tree.items():
    print(f"  {name:12s} {br.typename}")

ev = tree.arrays(library="ak")

# 2) variable length ---------------------------------------------------------
counts = np.bincount(ak.to_numpy(ev.nMuon))
print("\nnMuon  number_of_events")
for k, c in enumerate(counts):
    if c:
        print(f"  {k:3d}   {c}")
print("first 3 events, Muon_pt:", ak.to_list(ev.Muon_pt[:3]))

# 3) three ways to force a rectangular table ---------------------------------
padded = ak.to_numpy(ak.fill_none(ak.pad_none(ev.Muon_pt, 2, clip=True), np.nan))
print("\n(a) pad/clip to 2 muons -> shape", padded.shape,
      "| events that lose muons:", int(ak.sum(ev.nMuon > 2)))
print("(b) keep only events with nMuon == 2 ->", int(ak.sum(ev.nMuon == 2)), "rows")
print("(c) explode to one row per muon ->", int(ak.sum(ev.nMuon)), "rows")

# 4) invariant mass of the two leading muons, opposite charge ---------------
sel = ev[(ev.nMuon >= 2)]
idx = ak.argsort(sel.Muon_pt, ascending=False)
pt, eta, phi, q = (sel[b][idx][:, :2] for b in ("Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge"))
os_mask = (q[:, 0] * q[:, 1]) < 0
m2 = 2 * pt[:, 0] * pt[:, 1] * (np.cosh(eta[:, 0] - eta[:, 1]) - np.cos(phi[:, 0] - phi[:, 1]))
mass = ak.to_numpy(np.sqrt(m2[os_mask]))
in_window = (mass > 81) & (mass < 101)
print(f"\nopposite-charge pairs: {len(mass)}; with 81 < m < 101 GeV: {in_window.sum()}")
print(f"median mass in window: {np.median(mass[in_window]):.2f} GeV/c^2 (Z = 91.19)")

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(mass, bins=np.linspace(0, 150, 76), histtype="step", color="k")
ax.set_xlabel(r"dimuon invariant mass $m_{\mu\mu}$ [GeV/$c^2$]")
ax.set_ylabel("events / 2 GeV")
ax.set_title("SYNTHETIC CMS-like dimuon sample")
fig.tight_layout()
fig.savefig(OUT / "ch01_dimuon_mass.png", dpi=150)
print("saved outputs/ch01_dimuon_mass.png")
