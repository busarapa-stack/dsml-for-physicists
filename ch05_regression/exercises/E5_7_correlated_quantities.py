"""E5.7  When does ignoring correlations matter?  Q (no), amplitude at 5 s (yes, A-gamma),
and the time of a late crest of the cosine factor, t_k = (2 pi k - phi)/omega (yes, omega-phi)."""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
src = (HERE / '04_damped_oscillator.py').read_text()
exec(src[src.index('from scipy.optimize import curve_fit'):src.index('# --- numbers quoted in the text')])


def compare(name, value, J):
    naive = np.sqrt(np.sum(J**2 * np.diag(pcov)))
    full = np.sqrt(J @ pcov @ J)
    print(f"{name:28s} = {value:8.4f}: naive {naive:.4f}, full {full:.4f}, naive/full = {naive / full:.2f}")


print()
compare("Q = omega/(2 gamma)", omega / (2 * gamma), np.array([0, -omega / (2 * gamma**2), 1 / (2 * gamma), 0]))
compare("a(5 s) = A exp(-5 gamma) [m]", A * np.exp(-5 * gamma),
        np.array([np.exp(-5 * gamma), -5 * A * np.exp(-5 * gamma), 0, 0]))
for k in (1, 9):
    tk = (2 * np.pi * k - phi) / omega
    compare(f"crest time t_{k} [s]", tk, np.array([0, 0, -tk / omega, -1 / omega]))
print("-> Q: gamma and omega are almost uncorrelated, both formulas agree.\n"
      "   a(5 s) and crest times: the derivatives have the same sign and rho < 0 (omega-phi) or\n"
      "   opposite signs and rho > 0 (A-gamma), so the covariance term is negative and the naive\n"
      "   formula overestimates the uncertainty (table tab:correlation-sig)")
