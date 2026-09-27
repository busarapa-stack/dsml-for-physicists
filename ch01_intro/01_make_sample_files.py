"""Ch.1  Create small SYNTHETIC sample files in the four formats of Section 1.2.2.

Book references
    ssec:example, code:root-tree, code:ligo-hdf5, code:fits-header, code:matproj-json

The files mimic the *structure* of the real open datasets (CMS dimuon ROOT tree,
LIGO strain HDF5, SDSS FITS image/spectrum, Materials Project JSON) but the
numbers are generated here. They are small (a few MB) so every reader gets the
same result without downloading gigabytes. To open the real data, see optional/.

Output (in data/)
    dimuon_sample.root                     ROOT tree "Events" with jagged muon branches
    H-H1_SYNTH_4_V1-1126259446-32.hdf5     32 s of strain at 4096 Hz
    sdss_like_image_r.fits                 256 x 256 galaxy image, r band
    sdss_like_spectrum.fits                1D stellar spectrum (loglam, flux, ivar)
    mp_like_mp-149.json                    Materials-Project-style record for Si
"""
import json
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot
from astropy.io import fits

SEED = 2026
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

MUON_MASS = 0.1056584  # GeV/c^2
Z_MASS, Z_WIDTH = 91.19, 2.50  # GeV/c^2


# ---------------------------------------------------------------------------
# 1) ROOT: event-level data with a variable number of muons per event
# ---------------------------------------------------------------------------
def z_pair(n):
    """Two muons whose invariant mass follows a Breit-Wigner around the Z."""
    pts, etas, phis = [], [], []
    while len(pts) < n:
        m = Z_MASS + Z_WIDTH / 2 * np.tan(np.pi * (rng.random() - 0.5))
        m += rng.normal(0, 1.5)                       # detector resolution
        if not 60 < m < 120:
            continue
        eta = rng.uniform(-2.4, 2.4, 2)
        phi = rng.uniform(-np.pi, np.pi, 2)
        pt1 = rng.uniform(25, 60)
        # massless approximation: m^2 = 2 pt1 pt2 (cosh(d_eta) - cos(d_phi))
        pt2 = m**2 / (2 * pt1 * (np.cosh(eta[0] - eta[1]) - np.cos(phi[0] - phi[1])))
        if not 10 < pt2 < 150:
            continue
        pts.append([pt1, pt2]); etas.append(list(eta)); phis.append(list(phi))
    return pts, etas, phis


def make_root(n_events=5000, frac_signal=0.6):
    n_sig = int(frac_signal * n_events)
    pt, eta, phi, charge = [], [], [], []

    s_pt, s_eta, s_phi = z_pair(n_sig)
    for i in range(n_sig):
        q = rng.choice([-1, 1])
        p, e, f, c = s_pt[i], s_eta[i], s_phi[i], [q, -q]   # opposite charge
        # about 10 % of events carry one extra soft muon
        if rng.random() < 0.10:
            p = p + [rng.exponential(8) + 3]
            e = e + [rng.uniform(-2.4, 2.4)]
            f = f + [rng.uniform(-np.pi, np.pi)]
            c = c + [int(rng.choice([-1, 1]))]
        pt.append(p); eta.append(e); phi.append(f); charge.append(c)

    for _ in range(n_events - n_sig):                     # continuum background
        k = rng.choice([1, 2, 3], p=[0.2, 0.7, 0.1])
        pt.append(list(rng.exponential(12, k) + 3))
        eta.append(list(rng.uniform(-2.4, 2.4, k)))
        phi.append(list(rng.uniform(-np.pi, np.pi, k)))
        charge.append([int(x) for x in rng.choice([-1, 1], k)])

    order = rng.permutation(n_events)                     # mix signal and background
    pick = lambda a: [a[i] for i in order]
    f32 = lambda a: ak.values_astype(ak.Array(a), np.float32)   # real file uses float
    muon = ak.zip({
        "pt": f32(pick(pt)),
        "eta": f32(pick(eta)),
        "phi": f32(pick(phi)),
        "mass": ak.full_like(f32(pick(pt)), MUON_MASS),
        "charge": ak.values_astype(ak.Array(pick(charge)), np.int32),
    })

    path = DATA / "dimuon_sample.root"
    with uproot.recreate(path) as f:
        f.mktree("Events", {"Muon": muon.type.content},
                 counter_name=lambda c: "n" + c,
                 field_name=lambda outer, inner: f"{outer}_{inner}",
                 title="SYNTHETIC dimuon events (structure of CMS open data)")
        f["Events"].extend({"Muon": muon})
    return path


# ---------------------------------------------------------------------------
# 2) HDF5: 32 s strain time series at 4096 Hz with meta and quality groups
# ---------------------------------------------------------------------------
def make_hdf5():
    fs, duration, gps0 = 4096, 32, 1126259446
    n = fs * duration
    t = np.arange(n) / fs
    strain = rng.normal(0, 1e-21, n)                      # white noise, toy level

    # toy "chirp": frequency sweeps 35 -> 250 Hz during 0.2 s before t = 16.4 s
    tc = 16.4
    m = (t > tc - 0.2) & (t < tc)
    tau = tc - t[m]
    f_inst = 35 + (250 - 35) * (1 - tau / 0.2) ** 3
    phase = 2 * np.pi * np.cumsum(f_inst) / fs
    strain[m] += 1.0e-21 * (1 - tau / 0.2) * np.sin(phase)

    path = DATA / f"H-H1_SYNTH_4_V1-{gps0}-{duration}.hdf5"
    # Same layout as a real GWOSC/LOSC file (checked against
    # H-H1_LOSC_4_V1-1126256640-4096.hdf5). Note: there is no "SampleRate"
    # entry in meta/; the sampling interval is the attribute Xspacing of Strain.
    with h5py.File(path, "w") as h:
        meta = h.create_group("meta")
        meta["Description"] = b"SYNTHETIC strain for teaching (layout of a GWOSC file)"
        meta["DescriptionURL"] = b"https://gwosc.org/"
        meta["Detector"] = b"H1"
        meta["Duration"] = duration
        meta["GPSstart"] = gps0
        meta["Observatory"] = b"H"
        meta["Type"] = b"StrainTimeSeries"
        meta["UTCstart"] = b"2015-09-14T09:50:29"
        ds = h.create_group("strain").create_dataset("Strain", data=strain)
        ds.attrs.update({"Npoints": n, "Xlabel": "GPS time", "Xspacing": 1 / fs,
                         "Xstart": gps0, "Xunits": "second", "Ylabel": "Strain",
                         "Yunits": ""})
        q = h.create_group("quality")
        simple = q.create_group("simple")
        simple["DQShortnames"] = np.array([b"DATA", b"CBC_CAT1", b"CBC_CAT2", b"CBC_CAT3",
                                           b"BURST_CAT1", b"BURST_CAT2", b"BURST_CAT3"])
        simple["DQmask"] = np.full(duration, 127, dtype=np.uint32)   # all 7 bits = good
        inj = q.create_group("injections")
        inj["InjShortnames"] = np.array([b"NO_CBC_HW_INJ", b"NO_BURST_HW_INJ",
                                         b"NO_DETCHAR_HW_INJ", b"NO_CW_HW_INJ",
                                         b"NO_STOCH_HW_INJ"])
        inj["Injmask"] = np.full(duration, 31, dtype=np.uint32)      # 1 = no injection
    return path


# ---------------------------------------------------------------------------
# 3) FITS image: header (metadata) + 2D pixel array
# ---------------------------------------------------------------------------
def make_fits_image():
    ny = nx = 256
    y, x = np.mgrid[0:ny, 0:nx]
    x0, y0, q, pa = 128.0, 128.0, 0.6, np.deg2rad(30)
    xr = (x - x0) * np.cos(pa) + (y - y0) * np.sin(pa)
    yr = -(x - x0) * np.sin(pa) + (y - y0) * np.cos(pa)
    r = np.hypot(xr, yr / q)
    galaxy = 40.0 * np.exp(-r / 12.0)                     # exponential disk
    sky = 5.0
    img = rng.poisson(galaxy + sky).astype(np.float32) - sky

    hdr = fits.Header()
    hdr["RA"] = (179.689453125, "Right Ascension [deg]")
    hdr["DEC"] = (0.28563112068, "Declination [deg]")
    hdr["FILTER"] = ("r", "r-band")
    hdr["EXPTIME"] = (53.91, "exposure time [s]")
    hdr["BUNIT"] = ("counts", "pixel unit (sky subtracted)")
    hdr["COMMENT"] = "SYNTHETIC image for teaching; header keywords follow SDSS style"
    path = DATA / "sdss_like_image_r.fits"
    fits.PrimaryHDU(data=img, header=hdr).writeto(path, overwrite=True)
    return path


# ---------------------------------------------------------------------------
# 4) FITS spectrum: binary table with log10(wavelength), flux, inverse variance
# ---------------------------------------------------------------------------
def make_fits_spectrum():
    loglam = np.arange(np.log10(3800), np.log10(9200), 1e-4)   # SDSS-like sampling
    lam = 10**loglam                                            # Angstrom
    h, c, k, T = 6.626e-34, 2.998e8, 1.381e-23, 6000.0
    lam_m = lam * 1e-10
    bb = 1 / lam_m**5 / (np.exp(h * c / (lam_m * k * T)) - 1)
    flux = 50 * bb / bb.max()
    for line, depth, width in [(6563, 0.35, 6), (4861, 0.30, 5), (4340, 0.25, 5),
                               (5892, 0.20, 3)]:                # H-alpha/beta/gamma, Na D
        flux *= 1 - depth * np.exp(-0.5 * ((lam - line) / width) ** 2)
    sigma = 0.02 * flux.max() * np.ones_like(flux)
    flux_obs = flux + rng.normal(0, sigma)

    cols = fits.ColDefs([
        fits.Column(name="loglam", format="E", array=loglam),
        fits.Column(name="flux", format="E", array=flux_obs, unit="arbitrary"),
        fits.Column(name="ivar", format="E", array=1 / sigma**2),
    ])
    hdr = fits.Header()
    hdr["OBJTYPE"] = ("STAR", "object class")
    hdr["TEFF"] = (T, "temperature used to generate the continuum [K]")
    hdr["COMMENT"] = "SYNTHETIC spectrum for teaching (blackbody + 4 absorption lines)"
    path = DATA / "sdss_like_spectrum.fits"
    fits.HDUList([fits.PrimaryHDU(header=hdr),
                  fits.BinTableHDU.from_columns(cols, name="COADD")]).writeto(path, overwrite=True)
    return path


# ---------------------------------------------------------------------------
# 5) JSON: nested key-value record, as returned by the Materials Project API
# ---------------------------------------------------------------------------
def make_json():
    rec = {
        "_note": "Structure of an MP summary record. Fields set to null must be "
                 "fetched from the API (see optional/fetch_real_data.py).",
        "material_id": "mp-149",
        "formula_pretty": "Si",
        "symmetry": {"crystal_system": "Cubic", "symbol": "Fd-3m", "number": 227},
        "formation_energy_per_atom": 0.0,   # elemental reference, zero by definition
        "energy_above_hull": 0.0,           # ground state
        "is_stable": True,
        "band_gap": None,                   # eV, from DFT -> fetch from API
        "density": None,                    # g/cm^3       -> fetch from API
    }
    path = DATA / "mp_like_mp-149.json"
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    for fn in (make_root, make_hdf5, make_fits_image, make_fits_spectrum, make_json):
        p = fn()
        print(f"{p.name:42s} {p.stat().st_size / 1024:9.1f} kB")
