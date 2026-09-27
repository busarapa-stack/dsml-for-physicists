"""Reusable feature functions for AFM force-distance curves (Ch.3, E3.5; reused in Ch.4).

Every function takes one curve (a DataFrame with columns segment, distance_nm, force_nN)
whose baseline has already been removed, and returns a number WITH its unit in the name.
"""
import numpy as np
from scipy.signal import savgol_filter

_trapz = getattr(np, "trapezoid", None) or np.trapz        # numpy >= 2.0 / older


def remove_baseline(curve, far_nm=200.0):
    """Subtract the mean force where the tip is far from the surface (code:spectrum-clean)."""
    base = curve.loc[curve["distance_nm"] > far_nm, "force_nN"].mean()
    out = curve.copy()
    out["force_corrected"] = out["force_nN"] - base
    return out, base


def contact_slope_nN_per_nm(curve, f_min=0.5):
    """Slope dF/dz in the contact part of the APPROACH curve [nN/nm = N/m].
    Uses points where the corrected force is above f_min (clearly in contact)."""
    a = curve[(curve["segment"] == "approach") & (curve["force_corrected"] > f_min)]
    slope = np.polyfit(a["distance_nm"], a["force_corrected"], 1)[0]
    return -slope                                  # force rises as distance decreases


def max_adhesion_nN(curve, window=None):
    """Largest pulling (negative) force on the RETRACT curve [nN].
    window: optional Savitzky-Golay window, to show how smoothing biases the value."""
    f = curve.loc[curve["segment"] == "retract", "force_corrected"].values
    if window:
        f = savgol_filter(f, window, 3)
    return -f.min()


def adhesion_work_aJ(curve, far_nm=200.0, n_sigma=3.0):
    """Work of adhesion = area of the attractive part of the retract curve [nN nm = aJ].
    Only points clearly below zero (force < -n_sigma * noise) are counted; otherwise the
    negative half of the baseline noise over hundreds of nm adds a spurious 'work'."""
    r = curve[curve["segment"] == "retract"].sort_values("distance_nm")
    noise = curve.loc[curve["distance_nm"] > far_nm, "force_corrected"].std()
    f = r["force_corrected"].values
    f = np.where(f < -n_sigma * noise, f, 0.0)
    return -_trapz(f, r["distance_nm"].values)
