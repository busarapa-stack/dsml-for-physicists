"""Ch.4  Fit a Gaussian + baseline to an H-alpha line and turn its width into a temperature.

Book references: ssec:gaussian-fit, ex:gaussian-fit, code:gaussian-fit
Text: at T = 1e4 K, sigma_lambda ~ 0.020 nm and FWHM = 2.355 sigma ~ 0.047 nm.
The spectrum is SYNTHETIC (made with sigma for T = 1e4 K, see 01_make_data.py).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.constants as const

os.chdir(Path(__file__).resolve().parent)
spec = pd.read_csv('data/raw/halpha_spectrum.csv')
wavelength, intensity = spec['wavelength_nm'].values, spec['intensity'].values

# --- the estimate in the text -------------------------------------------------
lam0, T = 656.28, 1e4
m_H = const.m_p + const.m_e
sig = lam0 * np.sqrt(const.k * T / (m_H * const.c**2))
print(f"expected at T = 1e4 K: sigma = {sig:.4f} nm, FWHM = {2.355 * sig:.4f} nm "
      f"(2 sqrt(2 ln 2) = {2 * np.sqrt(2 * np.log(2)):.4f})")

# --- code:gaussian-fit ------------------------------------------------------
from scipy.optimize import curve_fit

def gaussian(x, A, mu, sigma, b):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + b

p0 = [intensity.max() - intensity.min(),
      wavelength[intensity.argmax()], 0.02, intensity.min()]
popt, pcov = curve_fit(gaussian, wavelength, intensity, p0=p0)
A, mu, sigma, b = popt
print(f"line centre  mu    = {mu:.4f} nm")
print(f"line width   sigma = {sigma:.4f} +/- {np.sqrt(pcov[2, 2]):.4f} nm")
print(f"FWHM = {2.355 * sigma:.4f} nm")

# --- invert: temperature from the fitted width -----------------------------------
T_fit = m_H * const.c**2 / const.k * (sigma / mu)**2
dT = 2 * T_fit * np.sqrt(pcov[2, 2]) / sigma
print(f"temperature from the width: T = {T_fit:.0f} +/- {dT:.0f} K (generated with 10000 K)")
