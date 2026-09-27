"""Ch.5  Five-fold cross-validation to choose the ridge penalty for a degree-10 polynomial.

Book references: ssec:cv-basics, ssec:hyper-PI, code:cv (reproduced line for line)
Uses the anharmonic-spring data of code:bv-scan (same seed, regenerated here).
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

os.chdir(Path(__file__).resolve().parent)
Path('outputs').mkdir(exist_ok=True)

# data exactly as in code:bv-scan
rng = np.random.default_rng(seed=2026)
k1, k3 = 5.0, 500.0
x_an = rng.uniform(-0.15, 0.15, 60)
F_an = k1 * x_an + k3 * x_an**3 + rng.normal(scale=0.05, size=x_an.size)
X_an = x_an.reshape(-1, 1)

# --- code:cv -----------------------------------------------------------------
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge

lambdas = np.logspace(-4, 2, 20)
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
cv_scores = []
for lam in lambdas:
    pipe = Pipeline([
        ('scale_x', StandardScaler()),
        ('poly',    PolynomialFeatures(degree=10, include_bias=False)),
        ('scale_f', StandardScaler()),
        ('reg',     Ridge(alpha=lam)),
    ])
    scores = cross_val_score(pipe, X_an, F_an, cv=kfold,
                             scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())
optimal_lambda = lambdas[np.argmin(cv_scores)]

# --- report ------------------------------------------------------------------
print("  lambda     CV MSE")
for lam, s in zip(lambdas, cv_scores):
    print(f"{lam:9.4f}  {s:.5f}")
print(f"optimal lambda = {optimal_lambda:.3f}, CV MSE = {min(cv_scores):.4f} N^2 (sigma^2 = 0.0025)")
flat = lambdas[np.array(cv_scores) < 1.1 * min(cv_scores)]
print(f"lambdas within 10 % of the minimum: {flat.min():.3f} to {flat.max():.3f}  (a flat valley)")

# the left side of the curve is not smooth: look at the individual folds
from sklearn.linear_model import LinearRegression
print("\nper-fold CV MSE at small lambda (and lambda = 0, plain OLS of degree 10):")
for lam in (0.0, 1e-4, 4.3e-4, 0.3):
    reg = LinearRegression() if lam == 0 else Ridge(alpha=lam)
    p_ = Pipeline([('scale_x', StandardScaler()),
                   ('poly', PolynomialFeatures(degree=10, include_bias=False)),
                   ('scale_f', StandardScaler()), ('reg', reg)])
    s_ = -cross_val_score(p_, X_an, F_an, cv=kfold, scoring='neg_mean_squared_error')
    print(f"  lambda = {lam:<7g} folds {np.round(s_, 4)}  mean {s_.mean():.4f}")
tr, va = next(iter(kfold.split(X_an)))
print(f"fold 1: validation x reaches {x_an[va].max():.3f} m, training x only up to {x_an[tr].max():.3f} m "
      "-> the degree-10 polynomial must extrapolate there; this fold makes the bump near lambda ~ 4e-4")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx(lambdas, cv_scores, 'o-'); ax.axvline(optimal_lambda, ls=':')
ax.axhline(0.0025, ls='--', color='k', label=r'$\sigma^2$')
ax.set_xlabel(r'$\lambda$ (alpha)'); ax.set_ylabel(r'5-fold CV MSE (N$^2$)'); ax.legend()
fig.tight_layout(); fig.savefig('outputs/ch05_ridge_cv.png', dpi=150)
print("saved outputs/ch05_ridge_cv.png")
