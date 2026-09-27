"""Ch.5  Numerical checks of statements in the text that have no listing of their own.

1. multicollinearity (ssec:phys-FE): correlation and VIF of polynomial features
2. log-transform notebox (ssec:phys-FE): unweighted straight line on ln y gives a biased tau
3. review question 2: slope SE for x in 0-5 cm vs 0-20 cm
4. review question 7: prediction uncertainty with and without the slope-intercept covariance
5. chi2_red notebox (ssec:chi2-red): how unlikely is chi2_red = 0.1 with 100 points?
"""
import os
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import curve_fit

os.chdir(Path(__file__).resolve().parent)


def vif(*cols):
    M = np.column_stack(cols)
    M = (M - M.mean(0)) / M.std(0)
    return np.diag(np.linalg.inv(np.corrcoef(M.T)))


# 1 ---------------------------------------------------------------------------
x = np.linspace(0.0, 0.20, 50)                     # stretches of code:hooke, all positive
z = (x - x.mean()) / x.std()
print("1. multicollinearity, x = 0-0.20 m (all positive)")
print(f"   corr(x, x^2) = {np.corrcoef(x, x**2)[0, 1]:.3f}, VIF = {vif(x, x**2)[0]:.0f}")
print(f"   corr(x, x^3) = {np.corrcoef(x, x**3)[0, 1]:.3f}, VIF = {vif(x, x**3)[0]:.0f}")
print(f"   x, x^2, x^3 together: VIF = {np.round(vif(x, x**2, x**3)).astype(int)}")
print(f"   standardized first: corr(z, z^2) = {np.corrcoef(z, z**2)[0, 1]:.3f}, "
      f"corr(z, z^3) = {np.corrcoef(z, z**3)[0, 1]:.3f}, VIF(z, z^2, z^3) = {np.round(vif(z, z**2, z**3), 1)}")

# 2 ---------------------------------------------------------------------------
rng = np.random.default_rng(2026)
t = np.linspace(0, 10, 50)
tau_true, y0, sig = 2.0, 100.0, 2.0                # constant noise in counts
tau_log, tau_cf, n_dropped = [], [], []
for _ in range(1000):
    y = y0 * np.exp(-t / tau_true) + rng.normal(scale=sig, size=t.size)
    ok = y > 0                                     # ln y impossible for y <= 0
    n_dropped.append((~ok).sum())
    slope = np.polyfit(t[ok], np.log(y[ok]), 1)[0]
    tau_log.append(-1 / slope)
    p, _ = curve_fit(lambda tt, a, b: a * np.exp(-tt / b), t, y, p0=[80, 1.5])
    tau_cf.append(p[1])
print("\n2. exponential decay, tau = 2.0 s, constant noise 2 counts, 1,000 data sets")
print(f"   points with y <= 0 that ln y must drop: {np.mean(n_dropped):.1f} per data set on average")
print(f"   unweighted line on ln y: tau = {np.mean(tau_log):.2f} +/- {np.std(tau_log):.2f} s (mean +/- spread)")
print(f"   curve_fit on y directly: tau = {np.mean(tau_cf):.3f} +/- {np.std(tau_cf):.3f} s")

# 3 ---------------------------------------------------------------------------
se = {w: 0.03 / np.sqrt(np.sum((xx - xx.mean())**2)) for w in (0.05, 0.20) for xx in [np.linspace(0, w, 50)]}
print(f"\n3. slope SE (sigma = 0.03 N, 50 points): 0-5 cm {se[0.05]:.3f} N/m, 0-20 cm {se[0.20]:.3f} N/m, "
      f"ratio {se[0.05] / se[0.20]:.1f}")

# 4 ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=2026)            # data of code:hooke
F_obs = 5.0 * x + rng.normal(scale=0.03, size=x.size)
fit = sm.OLS(F_obs, sm.add_constant(x)).fit()
C = fit.cov_params()
print(f"\n4. prediction y = c + m x0 (Hooke data), corr(c, m) = {C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]):.2f}")
for x0 in (0.0, 0.05, 0.10, 0.15, 0.20):
    J = np.array([1.0, x0])
    naive, full = np.sqrt(np.sum(J**2 * np.diag(C))), np.sqrt(J @ C @ J)
    print(f"   x0 = {x0:.2f} m: naive {naive:.4f} N, full {full:.4f} N, naive/full = {naive / full:.2f}")
print(f"   (mean of x = {x.mean():.2f} m)")

# 5 ---------------------------------------------------------------------------
nu = 100 - 2
print(f"\n5. chi2_red = 0.1 with 100 points (nu = {nu}): P(chi2_red <= 0.1) = {stats.chi2.cdf(0.1 * nu, nu):.0e}; "
      f"residuals smaller than the quoted sigma by sqrt(10) = {np.sqrt(10):.1f}")
