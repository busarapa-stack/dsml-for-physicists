"""Ch.9  AIC, BIC and blocked 10-fold CV for four oscillator models on the Chapter 5 data (tab:ic-damped);
likelihood-ratio test M2 -> M3; the many local minima of M4.

Book references: ssec:ic, tab:ic-damped, ssec:sig-testing (Wilks), review question 3
Data: code:damped-cf of Chapter 5 (200 points, sigma 0.05 m, seed 2026), executed from ../ch05_regression.
"""
import os
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
warnings.filterwarnings('ignore')
_s = (HERE.parent / 'ch05_regression' / '04_damped_oscillator.py').read_text()
exec(_s[_s.index('\n# --- code:damped-cf'):_s.index('\n# --- numbers quoted')])    # t, x_obs, sigma_x, ...


def M1(t, A, w, p): return A * np.cos(w * t + p)                                     # undamped
def M2(t, A, g, w, p): return A * np.exp(-g * t) * np.cos(w * t + p)                 # damped (true form)
def M3(t, A, g, w, p, c): return M2(t, A, g, w, p) + c                               # + zero offset
def M4(t, A, g, w, p, B, g2, w2, p2): return M2(t, A, g, w, p) + M2(t, B, g2, w2, p2)  # two damped terms


MODELS = {'M1': (M1, [1, 6, 0], ([0, 0, -np.pi], [10, 20, np.pi])),
          'M2': (M2, [1, .2, 6, 0], ([0, 0, 0, -np.pi], [10, 5, 20, np.pi])),
          'M3': (M3, [1, .2, 6, 0, 0], ([0, 0, 0, -np.pi, -1], [10, 5, 20, np.pi, 1])),
          'M4': (M4, None, ([0, 0, 0, -np.pi, 0, -5, 0, -np.pi], [10, 5, 20, np.pi, 10, 5, 20, np.pi]))}


def fit(f, p0, bounds, tt, xx, sig):
    return curve_fit(f, tt, xx, p0=p0, bounds=bounds, sigma=sig * np.ones(tt.size), absolute_sigma=True, maxfev=20000)


def fit_m4_multistart(tt, xx, sig):
    """M4 has many local minima: start from M2's best fit plus a small second term at ~1,200 starting points (99 values of w2 x 4 of g2 x 3 of p2)."""
    base = fit(M2, [1, .2, 6, 0], MODELS['M2'][2], tt, xx, sig)[0]
    found = []
    for w2 in np.arange(0.2, 20, 0.2):
        for g2 in (-0.3, -0.05, 0.05, 0.5):
            for p2 in (-2, 0, 2):
                try:
                    p, c = fit(M4, list(base) + [0.03, g2, w2, p2], MODELS['M4'][2], tt, xx, sig)
                except (RuntimeError, ValueError):
                    continue
                found.append((np.sum(((xx - M4(tt, *p)) / sig)**2), p, c))
    found.sort(key=lambda r: r[0])
    return found


def table(tt, xx, sig, label):
    n = tt.size
    rows, extra = {}, {}
    m4 = fit_m4_multistart(tt, xx, sig)
    for name, (f, p0, bounds) in MODELS.items():
        if name == 'M4':
            chi2, p, c = m4[0]
        else:
            p, c = fit(f, p0, bounds, tt, xx, sig)
            chi2 = np.sum(((xx - f(tt, *p)) / sig)**2)
        k = len(p)
        cv = {}
        for cv_name, splitter in [('blocked', KFold(10)), ('shuffled', KFold(10, shuffle=True, random_state=0))]:
            err = []
            for tr, te in splitter.split(tt):
                try:
                    pp, _ = curve_fit(f, tt[tr], xx[tr], p0=p, bounds=bounds, maxfev=20000)
                except RuntimeError:
                    pp = p
                err.append(np.mean((xx[te] - f(tt[te], *pp))**2))
            cv[cv_name] = np.sqrt(np.mean(err))
        rows[name] = (k, chi2, chi2 / (n - k), chi2 + 2 * k, chi2 + k * np.log(n), cv['blocked'], cv['shuffled'])
        extra[name] = (p, np.sqrt(np.diag(c)))
    print(f"\n{label}")
    print("model  k   chi2_min  chi2_red     AIC      BIC   CV blocked  CV shuffled")
    for name, r in rows.items():
        print(f"{name:5s} {r[0]:2d} {r[1]:9.1f} {r[2]:8.2f} {r[3]:8.1f} {r[4]:8.1f} {r[5]:10.4f} {r[6]:11.4f}")
    minima = np.unique(np.round([r[0] for r in m4], 1))
    print(f"M4 local minima found: {len(minima)} distinct chi2 values from {minima.min():.1f} to {minima[minima < minima.min() + 10].max():.1f} (+ worse ones)")
    p4 = extra['M4'][0]
    print(f"best M4 second term: B = {p4[4]:.3f} m, gamma2 = {p4[5]:.3f} 1/s, omega2 = {p4[6]:.2f} rad/s")
    print(f"M3 offset c = {extra['M3'][0][4]:.3f} +/- {extra['M3'][1][4]:.3f} m")
    return rows


if __name__ == '__main__':
    rows = table(t, x_obs, sigma_x, "Chapter 5 data: 200 points in 10 s, sigma 0.05 m (tab:ic-damped)")
    d23 = rows['M2'][1] - rows['M3'][1]
    print(f"\nWilks test M2 -> M3: delta chi2 = {d23:.1f}, 1 dof, p = {stats.chi2.sf(d23, 1):.2f}")
    print(f"AIC(M2) - AIC(M4) = {rows['M2'][3] - rows['M4'][3]:.1f}; BIC(M4) - BIC(M2) = {rows['M4'][4] - rows['M2'][4]:.1f}; "
          f"BIC(M3) - BIC(M2) = {rows['M3'][4] - rows['M2'][4]:.1f}")
    print(f"chi2 drop M2 -> M4: {rows['M2'][1] - rows['M4'][1]:.1f}; AIC penalty for 4 extra parameters 8, BIC penalty {4 * np.log(200):.1f}")
