"""Ch.6  The SDSS table used by code:sdss-cc and code:sdss-comp.

The book uses the 100,000-object table selected from SDSS DR17 and published for teaching as the
Kaggle "Stellar Classification Dataset - SDSS17" (fedesoriano).  Put the real file at
    data/raw/star_classification.csv
(see optional/README_sdss.md or run optional/fetch_sdss_skyserver.py) and this script only checks it.

If the real file is absent, this script writes a SYNTHETIC STAND-IN with the same columns and the same
class counts (59,445 galaxies, 21,594 stars, 18,961 quasars), so that every script of the chapter runs.
Its colours follow the physics described in the book only schematically (stellar locus, redder
galaxies, blue low-z quasars, z ~ 2.5-3 quasars on the stellar locus).  Numbers from the stand-in are
NOT results about the real sky; the book quotes no numbers from this table except the class fractions.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent)
RAW = Path('data/raw/star_classification.csv')
SOURCE = Path('data/raw/SOURCE.txt')
RAW.parent.mkdir(parents=True, exist_ok=True)


def make_standin(seed=2026):
    rng = np.random.default_rng(seed)
    n_gal, n_star, n_qso = 59445, 21594, 18961
    sc = lambda n, s: rng.normal(0, s, n)

    # stars: one temperature-like parameter t along the stellar locus (hot/blue t=0 -> cool/red t=1)
    t = rng.beta(2.0, 2.5, n_star)
    s_gr = -0.25 + 1.65 * t + sc(n_star, 0.05)
    # SDSS stellar locus: u-g stays near 1.0-1.2 for hot A/F stars (g-r < ~0.3), then climbs to ~2.6 for K/M
    s_ug = 1.05 + 1.6 * np.clip((s_gr - 0.25) / 0.95, 0, 1) + sc(n_star, 0.08)
    s_ri = 0.02 + 1.3 * t**2.2 + sc(n_star, 0.04)
    s_iz = 0.00 + 0.65 * t**2.2 + sc(n_star, 0.04)
    s_r = rng.uniform(14.0, 20.5, n_star)
    s_zred = rng.normal(0, 3e-4, n_star)

    # galaxies: red sequence and blue cloud, colours redden with redshift
    zg = np.clip(rng.gamma(3.0, 0.07, n_gal), 0.005, 1.0)
    red = rng.random(n_gal) < 0.55
    g_gr = np.where(red, 0.72 + 1.3 * zg, 0.42 + 0.9 * zg) + sc(n_gal, 0.10)
    g_ug = np.where(red, 1.75 + 0.8 * zg, 1.25 + 0.5 * zg) + sc(n_gal, 0.25)
    g_ri = 0.33 + 0.9 * zg + sc(n_gal, 0.08)
    g_iz = 0.22 + 0.35 * zg + sc(n_gal, 0.10)
    g_r = rng.uniform(15.0, 20.5, n_gal) + 2.0 * zg

    # quasars: blue in u-g until Lyman-alpha enters u (z > ~2.2), then u-g rises steeply
    zq = np.clip(np.concatenate([rng.normal(1.4, 0.6, int(0.7 * n_qso)),
                                 rng.normal(2.7, 0.35, n_qso - int(0.7 * n_qso))]), 0.1, 5.0)
    q_ug = 0.15 + 1.4 * np.clip(zq - 2.2, 0, None) + sc(n_qso, 0.18)
    q_gr = 0.15 + 0.25 * np.clip(zq - 2.4, 0, None) + 0.1 * np.sin(2.5 * zq) + sc(n_qso, 0.12)
    q_ri = 0.12 + 0.08 * np.sin(1.7 * zq) + sc(n_qso, 0.10)
    q_iz = 0.05 + 0.10 * np.cos(1.3 * zq) + sc(n_qso, 0.12)
    q_r = rng.uniform(17.0, 21.5, n_qso)

    def mags(ug, gr, ri, iz, r):
        g = r + gr; u = g + ug; i = r - ri; z = i - iz
        return u, g, r, i, z

    parts = []
    for cls, cols, zred in (('GALAXY', (g_ug, g_gr, g_ri, g_iz, g_r), zg),
                            ('STAR', (s_ug, s_gr, s_ri, s_iz, s_r), s_zred),
                            ('QSO', (q_ug, q_gr, q_ri, q_iz, q_r), zq)):
        u, g, r, i, z = mags(*cols)
        n = len(r)
        parts.append(pd.DataFrame({'alpha': rng.uniform(0, 360, n), 'delta': rng.uniform(-10, 70, n),
                                   'u': u, 'g': g, 'r': r, 'i': i, 'z': z,
                                   'class': cls, 'redshift': zred}))
    df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, 'obj_ID', np.arange(len(df)) + 1_000_000_000)
    df.loc[df.sample(1, random_state=seed).index, ['u', 'g', 'z']] = -9999.0   # one sentinel row, as in the real table
    return df.round({'u': 5, 'g': 5, 'r': 5, 'i': 5, 'z': 5, 'redshift': 6})


def is_synthetic(path):
    """The stand-in has no spec_obj_ID column; the Kaggle file and the SkyServer fetch both have it."""
    return 'spec_obj_ID' not in pd.read_csv(path, nrows=1).columns


if __name__ == '__main__':
    if not RAW.exists() or is_synthetic(RAW):          # a real file is never overwritten
        make_standin().to_csv(RAW, index=False)
        SOURCE.write_text("SYNTHETIC stand-in written by 05_sdss_data.py (not real SDSS data)\n")
    else:
        SOURCE.write_text("REAL SDSS table supplied by the user (Kaggle SDSS17 or optional/fetch_sdss_skyserver.py)\n")
    df = pd.read_csv(RAW)
    print(f"source: {SOURCE.read_text().strip()}")
    print(f"{len(df)} rows; columns: {list(df.columns)}")
    frac = df['class'].value_counts(normalize=True)
    print("class fractions: " + ", ".join(f"{k} {v:.3f}" for k, v in frac.items()))
    bad = (df[['u', 'g', 'r', 'i', 'z']] < -100).any(axis=1).sum()
    print(f"rows with a sentinel magnitude (< -100): {bad}")
