"""Ch.1  One table for all sample files: format, modality, shape, size, metadata.

Book references
    ssec:synthesis, tab:modalities, exercise ex:matching

Run after 01-05. Writes outputs/ch01_format_summary.csv.
"""
import csv
import json
from pathlib import Path

import h5py
import uproot
from astropy.io import fits

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "outputs"
OUT.mkdir(exist_ok=True)

rows = []

p = DATA / "dimuon_sample.root"
t = uproot.open(p)["Events"]
rows.append(dict(file=p.name, format="ROOT", modality="tabular (jagged)",
                 shape=f"{t.num_entries} events x {len(t.keys())} branches",
                 metadata_items="1 (tree title)"))

p = next(DATA.glob("H-H1_*.hdf5"))
with h5py.File(p) as h:
    n_meta = len(h["meta"]) + sum(len(o.attrs) for o in
                                  [h[k] for k in ("strain/Strain", "quality/injections")])
    rows.append(dict(file=p.name, format="HDF5", modality="time series",
                     shape=f"{h['strain/Strain'].shape[0]} samples",
                     metadata_items=n_meta))

p = DATA / "sdss_like_image_r.fits"
with fits.open(p) as hd:
    rows.append(dict(file=p.name, format="FITS", modality="image",
                     shape="x".join(map(str, hd[0].data.shape)),
                     metadata_items=len(hd[0].header)))

p = DATA / "sdss_like_spectrum.fits"
with fits.open(p) as hd:
    rows.append(dict(file=p.name, format="FITS", modality="spectrum",
                     shape=f"{len(hd[1].data)} wavelengths",
                     metadata_items=len(hd[0].header) + len(hd[1].header)))

p = DATA / "mp_like_mp-149.json"
rec = json.loads(p.read_text(encoding="utf-8"))
rows.append(dict(file=p.name, format="JSON", modality="tabular (one row per material)",
                 shape=f"{sum(1 for k in rec if not k.startswith('_'))} top-level fields",
                 metadata_items="-"))

for r in rows:
    r["size_kB"] = round((DATA / r["file"]).stat().st_size / 1024, 1)

cols = ["file", "format", "modality", "shape", "size_kB", "metadata_items"]
w = [max(len(str(r[c])) for r in rows + [dict(zip(cols, cols))]) for c in cols]
print("  ".join(c.ljust(x) for c, x in zip(cols, w)))
for r in rows:
    print("  ".join(str(r[c]).ljust(x) for c, x in zip(cols, w)))

with open(OUT / "ch01_format_summary.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=cols)
    wr.writeheader(); wr.writerows(rows)
print("\nsaved outputs/ch01_format_summary.csv")
