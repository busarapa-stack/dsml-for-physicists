"""Ch.1 (optional)  Check the REAL files downloaded by fetch_real_data.py against
the statements made in Section 1.2.2 of the book. Runs offline.

    python optional/fetch_real_data.py      # first, needs internet (once)
    python optional/check_real_vs_book.py   # then, offline

Each check prints the book statement, the value found in the real data and
OK / DIFFERS. Parts whose files are missing are skipped.
"""
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
REAL = HERE / "data" / "real"

GW150914_FILE_START = 1126259446       # start of the 32 s file named in code:ligo-hdf5
SI_A_EXP = 5.431                       # Angstrom, experimental lattice constant (300 K)
SI_GAP_EXP = 1.12                      # eV, experimental band gap (300 K)
SI_RHO_EXP = 2.329                     # g/cm^3, experimental density
U = 1.66053907e-24                     # g, atomic mass unit
M_SI = 28.0855                         # g/mol

n_ok = n_diff = 0


def report(statement, found, ok):
    global n_ok, n_diff
    n_ok += ok
    n_diff += not ok
    print(f"  [{'OK' if ok else 'DIFFERS'}] {statement}\n        found: {found}")


def ligo():
    import h5py
    files = sorted(REAL.glob("H-H1_*.hdf5"))
    if not files:
        print("  (no LIGO file; run fetch_real_data.py ligo)"); return
    with h5py.File(files[0]) as h:
        meta = sorted(h["meta"].keys())
        s = h["strain/Strain"]
        fs = int(round(1 / s.attrs["Xspacing"]))
        gps0 = int(h["meta/GPSstart"][()])
        i0 = (GW150914_FILE_START - gps0) * fs
        seg = s[i0:i0 + 32 * fs]
        dq = h["quality/simple/DQmask"][()]
        j0 = GW150914_FILE_START - gps0
        groups = sorted(h["quality"].keys())
    report("meta/ says which detector and when (code:ligo-hdf5)",
           ", ".join(meta), "Detector" in meta and "GPSstart" in meta)
    report("sampling rate is the Strain attribute Xspacing, not a meta/ entry",
           f"Xspacing -> {fs} Hz; 'SampleRate' in meta: {'SampleRate' in meta}",
           fs == 4096 and "SampleRate" not in meta)
    report("32 s x 4096 Hz = 131,072 samples", f"{len(seg):,} samples from GPS "
           f"{GW150914_FILE_START}", len(seg) == 131_072)
    report("quality/ has per-second quality flags and injections",
           f"quality groups: {groups}; DQmask over the 32 s: {sorted(set(dq[j0:j0+32].tolist()))}",
           {"simple", "injections"} <= set(groups))
    report("raw strain is far above 1e-21 before whitening (noise dominated)",
           f"max |h| in the 32 s = {np.abs(seg).max():.2e}", np.abs(seg).max() > 1e-20)


def cms():
    p = REAL / "cms_dimuon_mass_first100k.npy"
    if not p.exists():
        print("  (no CMS file; run fetch_real_data.py cms)"); return
    m = np.load(p)
    for name, lo, hi, true in [("J/psi", 2.9, 3.3, 3.097), ("Upsilon(1S)", 9.2, 9.7, 9.460),
                               ("Z", 81, 101, 91.19)]:
        w = m[(m > lo) & (m < hi)]
        med = np.median(w)
        report(f"{name} peak visible in the dimuon mass (PDG {true} GeV)",
               f"{len(w)} pairs in {lo}-{hi} GeV, median {med:.3f} GeV",
               abs(med - true) / true < 0.02)


def mp():
    pj = REAL / "mp-149_real.json"
    if not pj.exists():
        print("  (no Materials Project file; run fetch_real_data.py mp)"); return
    rec = json.loads(pj.read_text(encoding="utf-8"))
    report("formation energy of a pure element is 0 (reference state)",
           rec["formation_energy_per_atom"], rec["formation_energy_per_atom"] == 0)
    report("energy_above_hull == 0 <=> is_stable",
           f"{rec['energy_above_hull']} / {rec['is_stable']}",
           (rec["energy_above_hull"] == 0) == rec["is_stable"])
    low = 1 - rec["band_gap"] / SI_GAP_EXP
    report("GGA band gap lower than experiment by roughly 40 % (misconception box)",
           f"{rec['band_gap']:.3f} eV vs {SI_GAP_EXP} eV -> {100 * low:.0f} % lower",
           0.3 < low < 0.6)

    pc = REAL / "mp-149_Si.cif"
    if not pc.exists():
        return
    txt = pc.read_text()
    get = lambda k: float(re.search(rf"{k}\s+([\d.]+)", txt).group(1))
    a_p, vol = get("_cell_length_a"), get("_cell_volume")
    n_at = int(re.search(r"_chemical_formula_sum\s+Si(\d+)", txt).group(1))
    a_conv = a_p * np.sqrt(2)                   # fcc primitive (60 deg) -> cubic cell
    d_nn = a_conv * np.sqrt(3) / 4
    rho = n_at * M_SI * U / (vol * 1e-24)
    report("density field agrees with the stored crystal structure",
           f"from CIF {rho:.4f} g/cm^3, JSON {rec['density']:.4f} g/cm^3",
           abs(rho - rec["density"]) < 1e-3)
    report("DFT (GGA) lattice slightly larger than experiment -> density slightly lower",
           f"a = {a_conv:.3f} A vs {SI_A_EXP} A ({100 * (a_conv / SI_A_EXP - 1):+.2f} %), "
           f"d_nn = {d_nn:.3f} A vs {SI_A_EXP * np.sqrt(3) / 4:.3f} A, "
           f"rho {rec['density']:.3f} vs {SI_RHO_EXP} g/cm^3",
           a_conv > SI_A_EXP and rec["density"] < SI_RHO_EXP)


if __name__ == "__main__":
    for title, fn in [("LIGO GW150914", ligo), ("CMS dimuon", cms), ("Materials Project mp-149", mp)]:
        print(f"\n===== {title} =====")
        try:
            fn()
        except Exception as e:
            print(f"  [skipped] {type(e).__name__}: {e}")
    print(f"\n{n_ok} OK, {n_diff} DIFFERS")
