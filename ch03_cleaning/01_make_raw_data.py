"""Ch.3  Create the SYNTHETIC raw data used by every script of this chapter.

The book refers to files such as data/raw/galaxy10_metadata.csv and data/raw/afm_spectrum.csv.
They are teaching files built here with a fixed seed; the numbers are NOT measurements.
  * galaxy table: Galaxy10 DECaLS class names, but redshift and magnitudes are simulated.
  * AFM files: simulated curves (the author's own AFM measurements are not published here).
  * 4-lepton events: simulated H -> ZZ* -> 4l signal plus a smooth background.

Output (data/raw/)
  galaxy10_metadata.csv      2,000 galaxies: image_id, galaxy_class, redshift, r_mag, g_mag
  galaxy10_image_index.csv   image_id -> image file (some ids missing, some extra)
  galaxy10_truth.csv         the true redshift of every galaxy (answer key, for E3.4 only)
  galaxy10_messy.csv         ~500 rows with NaN, -999 sentinels and outliers (for E3.2)
  galaxy_images.npy          4 synthetic images, shape (4, 256, 256, 3), 2 hot pixels in image 0
  afm_timeseries.csv         force signal while scanning: drift + spikes + noise
  afm_spectrum.csv           one force-distance curve (approach + retract) with baseline shift
  afm_curves.csv             15 force-distance curves with different stiffness/adhesion (E3.5)
  afm_curves_truth.csv       the parameters used to make them (answer key)
  higgs_4l_events.csv        E, px, py, pz of four leptons per event [GeV]
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
rng = np.random.default_rng(SEED)
RAW = Path(__file__).resolve().parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

CLASSES = ["Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
           "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
           "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge"]


# --------------------------------------------------------------- galaxies
def make_galaxies(n=2000):
    cls = rng.choice(len(CLASSES), n, p=[.06, .10, .15, .15, .06, .11, .10, .15, .06, .06])
    z_true = np.clip(rng.gamma(3.0, 0.035, n), 0.005, 0.40)
    # toy magnitude-redshift relation (inverse of photo_z_estimate in the book) + scatter
    r_mag = 15.0 + z_true / 0.05 + rng.normal(0, 0.6, n)
    color = np.array([0.7, 0.7, 0.9, 0.85, 0.8, 0.75, 0.6, 0.55, 0.65, 0.8])[cls]
    g_mag = r_mag + color + rng.normal(0, 0.1, n)
    # redshift missing more often for faint galaxies (MAR on r_mag), ~15 % overall
    p_miss = 1 / (1 + np.exp(-(r_mag - 18.9) / 0.6))
    missing = rng.random(n) < p_miss
    df = pd.DataFrame({
        "image_id": [f"galaxy_{i:05d}" for i in range(n)],
        "galaxy_class": [CLASSES[c] for c in cls],
        "redshift": np.where(missing, np.nan, np.round(z_true, 4)),
        "r_mag": np.round(r_mag, 3),
        "g_mag": np.round(g_mag, 3).astype(object),
    })
    # five bright foreground stars that slipped into the catalogue (r < 8)
    stars = rng.choice(n, 5, replace=False)
    df.loc[stars, "r_mag"] = np.round(rng.uniform(6.0, 7.8, 5), 3)
    df.loc[stars, "redshift"] = 0.0
    # three g_mag values typed as text in the source file -> the column is read as text
    bad = rng.choice(np.setdiff1d(np.arange(n), stars), 3, replace=False)
    df.loc[bad, "g_mag"] = "--"
    df.to_csv(RAW / "galaxy10_metadata.csv", index=False)
    pd.DataFrame({"image_id": df["image_id"], "z_true": np.round(z_true, 4),
                  "is_star": np.isin(np.arange(n), stars)}).to_csv(RAW / "galaxy10_truth.csv",
                                                                   index=False)

    # image index: 20 galaxies have no image file, 15 image files have no metadata row
    keep = np.sort(rng.choice(n, n - 20, replace=False))
    ids = [f"galaxy_{i:05d}" for i in keep] + [f"galaxy_{i:05d}" for i in range(n, n + 15)]
    pd.DataFrame({"image_id": ids,
                  "image_file": [f"images/{i}.png" for i in ids]}).to_csv(
        RAW / "galaxy10_image_index.csv", index=False)
    return df, z_true


def make_messy(df, n=500):
    m = df.sample(n, random_state=SEED).reset_index(drop=True).copy()
    m["g_mag"] = pd.to_numeric(m["g_mag"], errors="coerce")
    idx = rng.permutation(n)
    m.loc[idx[:20], "redshift"] = -999          # sentinel: "could not be measured"
    m.loc[idx[20:25], "r_mag"] = -999
    m.loc[idx[25:31], "r_mag"] = np.round(rng.uniform(25.5, 28.0, 6), 3)   # too faint
    m.loc[idx[31:33], "redshift"] = np.round(-rng.uniform(0.01, 0.05, 2), 4)  # sign typo
    m.to_csv(RAW / "galaxy10_messy.csv", index=False)


def make_images(n=4, size=256):
    y, x = np.mgrid[0:size, 0:size]
    imgs = np.zeros((n, size, size, 3), dtype=np.float32)
    for k in range(n):
        x0, y0 = 128 + rng.normal(0, 3, 2)
        q, pa = rng.uniform(0.5, 1.0), rng.uniform(0, np.pi)
        xr = (x - x0) * np.cos(pa) + (y - y0) * np.sin(pa)
        yr = -(x - x0) * np.sin(pa) + (y - y0) * np.cos(pa)
        r = np.hypot(xr, yr / q)
        for b, amp in enumerate((0.6, 1.0, 1.3)):                  # g, r, z bands
            bulge = amp * 60 * np.exp(-7.67 * ((r / 6) ** 0.25 - 1)) / np.exp(7.67)
            disk = amp * 8 * np.exp(-r / 25)
            imgs[k, :, :, b] = bulge + disk + rng.normal(0, 0.5, (size, size))
    imgs[0, 40, 200, 0] = 60.0        # two hot pixels in image 0, band 0
    imgs[0, 210, 35, 0] = 45.0
    np.save(RAW / "galaxy_images.npy", imgs)


# --------------------------------------------------------------- AFM
def afm_curve(k_sample, adhesion_nN, z_contact=20.0, snap=15.0, offset=0.8, noise=0.05,
              n=301):
    """One approach + retract force-distance curve [nN vs nm].
    distance_nm is the piezo height above the contact point + z_contact."""
    z = np.linspace(300.0, 0.0, n)
    appr = np.where(z < z_contact, k_sample * (z_contact - z), 0.0)
    z_r = z[::-1]
    retr = np.where(z_r < z_contact, k_sample * (z_contact - z_r), 0.0)
    # adhesion: the tip stays stuck and is pulled down to -adhesion before snapping off
    stick = (z_r >= z_contact) & (z_r <= z_contact + snap)
    retr = np.where(stick, -adhesion_nN * (z_r - z_contact) / snap, retr)
    d = np.concatenate([z, z_r])
    f = np.concatenate([appr, retr]) + offset + rng.normal(0, noise, 2 * n)
    seg = ["approach"] * n + ["retract"] * n
    return pd.DataFrame({"segment": seg, "distance_nm": np.round(d, 3),
                         "force_nN": np.round(f, 4)})


def make_afm():
    # time series: 2 s at 1 kHz while the tip scans a surface
    t = np.arange(0, 2.0, 0.001)
    topo = 0.3 * np.sin(2 * np.pi * 1.5 * t)
    drift = 0.4 * t                                  # slow thermal drift [nN/s]
    f = 2.0 + topo + drift + rng.normal(0, 0.03, t.size)
    spikes = rng.choice(t.size, 15, replace=False)
    f[spikes] += rng.choice([-1, 1], 15) * rng.uniform(1.0, 2.5, 15)
    pd.DataFrame({"time_s": np.round(t, 4), "force_nN": np.round(f, 4)}).to_csv(
        RAW / "afm_timeseries.csv", index=False)
    pd.DataFrame({"spike_index": np.sort(spikes)}).to_csv(RAW / "afm_timeseries_spikes.csv",
                                                           index=False)

    afm_curve(k_sample=0.25, adhesion_nN=3.0).to_csv(RAW / "afm_spectrum.csv", index=False)

    rows, truth = [], []
    for cid in range(15):
        k_s = rng.uniform(0.1, 0.5)                  # nN/nm
        adh = rng.uniform(1.0, 5.0)                  # nN
        off = rng.normal(0.5, 0.3)
        c = afm_curve(k_s, adh, offset=off)
        c.insert(0, "curve_id", cid)
        rows.append(c)
        truth.append({"curve_id": cid, "slope_nN_per_nm": k_s, "adhesion_nN": adh,
                      "work_aJ": 0.5 * adh * 15.0, "baseline_nN": off})
    pd.concat(rows).to_csv(RAW / "afm_curves.csv", index=False)
    pd.DataFrame(truth).round(4).to_csv(RAW / "afm_curves_truth.csv", index=False)


# --------------------------------------------------------------- 4-lepton events
def two_body(M, m1, m2, n):
    """Momenta of two daughters in the parent rest frame, isotropic."""
    p = np.sqrt((M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)) / (2 * M)
    cth = rng.uniform(-1, 1, n); phi = rng.uniform(0, 2 * np.pi, n)
    sth = np.sqrt(1 - cth**2)
    v = np.stack([p * sth * np.cos(phi), p * sth * np.sin(phi), p * cth], axis=1)
    E1 = np.sqrt(p**2 + m1**2); E2 = np.sqrt(p**2 + m2**2)
    return np.column_stack([E1, v]), np.column_stack([E2, -v])


def boost(P, beta):
    """Lorentz boost of four-vectors P (n,4) by velocity vectors beta (n,3)."""
    b2 = np.sum(beta**2, axis=1)
    g = 1 / np.sqrt(1 - b2)
    bp = np.sum(beta * P[:, 1:], axis=1)
    E = g * (P[:, 0] + bp)
    coef = np.where(b2 > 0, (g - 1) * bp / np.where(b2 > 0, b2, 1), 0.0) + g * P[:, 0]
    return np.column_stack([E, P[:, 1:] + coef[:, None] * beta])


def make_4l(n_sig=400, n_bkg=1600):
    M = np.concatenate([rng.normal(125.0, 1.8, n_sig),              # Higgs + resolution
                        70 + rng.gamma(2.5, 45.0, n_bkg)])            # smooth background
    n = M.size
    mZ1 = np.minimum(rng.normal(91.19, 2.5, n), M - 13)
    mZ2 = np.clip(rng.uniform(12, 60, n), 12, M - mZ1 - 1)
    Z1, Z2 = two_body(M, mZ1, mZ2, n)
    leps = []
    for Z, mZ in ((Z1, mZ1), (Z2, mZ2)):
        l1, l2 = two_body(mZ, 0.0, 0.0, n)
        beta = Z[:, 1:] / Z[:, [0]]
        leps += [boost(l1, beta), boost(l2, beta)]
    # boost the whole system along the beam (z) and give it a little transverse kick
    beta_lab = np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n),
                                rng.uniform(-0.7, 0.7, n)])
    leps = [boost(l, beta_lab) for l in leps]
    order = rng.permutation(n)
    cols = {}
    for i, l in enumerate(leps, start=1):
        for j, c in enumerate(("E", "px", "py", "pz")):
            cols[f"{c}{i}"] = np.round(l[order, j], 4)
    ev = pd.DataFrame(cols)
    ev.insert(0, "is_signal", (np.arange(n) < n_sig)[order])
    ev.to_csv(RAW / "higgs_4l_events.csv", index=False)


if __name__ == "__main__":
    df, _ = make_galaxies()
    make_messy(df)
    make_images()
    make_afm()
    make_4l()
    for p in sorted(RAW.iterdir()):
        print(f"{p.name:30s} {p.stat().st_size / 1024:8.1f} kB")
