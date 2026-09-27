"""Ch.1 (optional)  Open the REAL open datasets described in Section 1.2.2.

Needs an internet connection. Each part runs independently; a failure in one
part (no network, missing package, no API key) is reported and skipped.

    python optional/fetch_real_data.py            # all parts
    python optional/fetch_real_data.py ligo cms   # selected parts
    python optional/fetch_real_data.py cmsplot    # plot after cms (offline)
    python optional/check_real_vs_book.py         # compare real data with the book (offline)

Extra packages:  pip install gwosc mp-api aiohttp requests

Parts
    ligo  GW150914, H1, 4096 Hz from GWOSC; takes the shortest file   CC BY 4.0
          listed (in 2026 only the 4096 s O1 file, >100 MB)
    cms   first 100,000 events of the CMS 2012 dimuon sample,
          read directly over HTTP with uproot (file itself is ~2 GB) CC0
    mp    record mp-149 (Si) from the Materials Project API, saved as
          data/real/mp-149_real.json and data/real/mp-149_Si.cif;
          set the environment variable MP_API_KEY first
          (free key from https://materialsproject.org)             CC BY 4.0

Tested on 2026-09-26 (macOS, Python 3.14): ligo, cms, cmsplot and mp OK; results in
optional/expected_output_real.txt. The numbers in the book and in
expected_output.txt come from the synthetic files of 01_make_sample_files.py.
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REAL = HERE / "data" / "real"
REAL.mkdir(parents=True, exist_ok=True)

CMS_URL = ("https://opendata.cern.ch/eos/opendata/cms/derived-data/"
           "AOD2NanoAODOutreachTool/Run2012BC_DoubleMuParked_Muons.root")
CMS_XROOTD = ("root://eospublic.cern.ch//eos/opendata/cms/derived-data/"
              "AOD2NanoAODOutreachTool/Run2012BC_DoubleMuParked_Muons.root")


def _duration(url):
    """Duration in seconds from a GWOSC file name like ...-1126259447-32.hdf5"""
    m = re.search(r"-(\d+)\.hdf5", url)
    return int(m.group(1)) if m else 10**9


def ligo():
    import h5py
    import numpy as np
    from gwosc.datasets import event_gps
    from gwosc.locate import get_event_urls, get_urls
    urls = [u for u in get_event_urls("GW150914", detector="H1") if ".hdf5" in u]
    if not urls:  # fall back: ask by GPS time instead of by event name
        gps = event_gps("GW150914")
        urls = [u for u in get_urls("H1", gps - 16, gps + 16) if ".hdf5" in u]
    if not urls:
        raise RuntimeError("GWOSC listed no HDF5 file for GW150914 H1")
    print("files listed by GWOSC:")
    for u in urls:
        print("  ", u)
    url = min(urls, key=_duration)          # the shortest file (32 s) is enough
    dest = REAL / Path(url.split("?")[0]).name
    if not dest.exists():
        print("downloading", url)
        urllib.request.urlretrieve(url, dest)
    with h5py.File(dest) as h:
        h.visit(print)
        fs = int(round(1 / h["strain/Strain"].attrs["Xspacing"]))   # not in meta/
        s = h["strain/Strain"][:]
        print(f"samples {len(s):,} = {len(s) / fs:.0f} s x {fs} Hz; max |h| = {abs(s).max():.2e}")
        # cut the 32 s window of the file named in code:ligo-hdf5 (GPS 1126259446)
        i0 = (1126259446 - int(h["meta/GPSstart"][()])) * fs
        if 0 <= i0 <= len(s) - 32 * fs:
            seg = s[i0:i0 + 32 * fs]
            np.save(REAL / "gw150914_H1_32s_from_1126259446.npy", seg)
            print(f"32 s window from GPS 1126259446: {len(seg):,} samples, "
                  f"max |h| = {abs(seg).max():.2e}; saved data/real/gw150914_H1_32s_from_1126259446.npy")
        print("note: the raw strain is dominated by low-frequency noise; the\n"
              "      ~1e-21 signal appears only after whitening and band-pass (Ch.4)")


def cms(n=100_000):
    import awkward as ak
    import numpy as np
    import uproot
    try:
        tree = uproot.open(CMS_URL)["Events"]
    except Exception as e:  # fall back to XRootD (needs: pip install fsspec-xrootd)
        print("HTTP failed:", e, "\ntrying XRootD ...")
        tree = uproot.open(CMS_XROOTD)["Events"]
    tree.show()
    ev = tree.arrays(["nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge"],
                     entry_stop=n)
    two = ev[(ev.nMuon == 2) & (ak.sum(ev.Muon_charge, axis=1) == 0)]
    pt, eta, phi = two.Muon_pt, two.Muon_eta, two.Muon_phi
    m = np.sqrt(2 * pt[:, 0] * pt[:, 1] *
                (np.cosh(eta[:, 0] - eta[:, 1]) - np.cos(phi[:, 0] - phi[:, 1])))
    print(f"{len(m)} opposite-charge pairs in the first {n:,} events")
    np.save(REAL / "cms_dimuon_mass_first100k.npy", ak.to_numpy(m))
    print("saved data/real/cms_dimuon_mass_first100k.npy"
          " (plot on a log axis to see J/psi, Upsilon and Z)")


def mp():
    from mp_api.client import MPRester
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("set MP_API_KEY first")
    fields = ["material_id", "formula_pretty", "symmetry", "formation_energy_per_atom",
              "energy_above_hull", "is_stable", "band_gap", "density"]
    with MPRester(key) as mpr:
        doc = mpr.materials.summary.search(material_ids=["mp-149"],
                                           fields=fields + ["structure"])[0]
    for f in fields:
        print(f"{f:28s} {getattr(doc, f)}")

    # save the record as JSON, same layout as data/mp_like_mp-149.json
    sym = doc.symmetry
    rec = {
        "_source": "Materials Project API (CC BY 4.0), fetched "
                   + __import__("datetime").date.today().isoformat(),
        "material_id": str(doc.material_id),
        "formula_pretty": doc.formula_pretty,
        "symmetry": {"crystal_system": str(getattr(sym.crystal_system, "value", sym.crystal_system)),
                     "symbol": sym.symbol, "number": sym.number},
        "formation_energy_per_atom": doc.formation_energy_per_atom,
        "energy_above_hull": doc.energy_above_hull,
        "is_stable": doc.is_stable,
        "band_gap": doc.band_gap,
        "density": doc.density,
    }
    out = REAL / "mp-149_real.json"
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved data/real/mp-149_real.json")

    # save the relaxed crystal structure (for the graph modality, 06_crystal_graph.py)
    cif = REAL / "mp-149_Si.cif"
    doc.structure.to(filename=str(cif))
    lat = doc.structure.lattice
    print(f"saved data/real/mp-149_Si.cif  ({len(doc.structure)} atoms in cell, "
          f"a = {lat.a:.3f} A, angles {lat.alpha:.1f}/{lat.beta:.1f}/{lat.gamma:.1f} deg)")


def cmsplot():
    """Plot the dimuon mass saved by the cms part (no download needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    m = np.load(REAL / "cms_dimuon_mass_first100k.npy")
    bins = np.logspace(np.log10(0.25), np.log10(300), 300)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(m, bins=bins, histtype="step", color="k")
    ax.set_xscale("log"); ax.set_yscale("log")
    for name, mass in [("J/psi", 3.097), ("Upsilon", 9.46), ("Z", 91.19)]:
        ax.axvline(mass, ls=":", color="gray")
        ax.text(mass, ax.get_ylim()[1] * 0.5, name, rotation=90, va="top", ha="right")
    ax.set_xlabel(r"dimuon invariant mass [GeV/$c^2$]")
    ax.set_ylabel("events / bin")
    ax.set_title(f"CMS Open Data 2012, first 100,000 events ({len(m):,} pairs)")
    fig.tight_layout()
    out = HERE / "outputs" / "ch01_real_cms_dimuon_mass.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    for lo, hi, name in [(2.9, 3.3, "J/psi"), (9.2, 9.7, "Upsilon(1S)"), (81, 101, "Z")]:
        print(f"{name:12s} {lo:>5}-{hi:<5} GeV : {((m > lo) & (m < hi)).sum():6d} pairs")
    print("saved outputs/ch01_real_cms_dimuon_mass.png")


PARTS = {"ligo": ligo, "cms": cms, "cmsplot": cmsplot, "mp": mp}

if __name__ == "__main__":
    chosen = sys.argv[1:] or list(PARTS)
    for name in chosen:
        print(f"\n===== {name} =====")
        try:
            PARTS[name]()
        except Exception as e:
            print(f"[skipped] {type(e).__name__}: {e}")
