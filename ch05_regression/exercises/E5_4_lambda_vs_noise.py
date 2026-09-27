"""E5.4  Optimal ridge lambda (code:cv) for noise 0.05 N and 0.15 N."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

os.chdir(Path(__file__).resolve().parent.parent)
lambdas = np.logspace(-4, 2, 20)
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

fig, ax = plt.subplots(figsize=(6, 4))
for noise in (0.05, 0.15):
    rng = np.random.default_rng(seed=2026)                 # same draws as code:bv-scan
    x_an = rng.uniform(-0.15, 0.15, 60)
    F_an = 5.0 * x_an + 500.0 * x_an**3 + rng.normal(scale=noise, size=x_an.size)
    X_an = x_an.reshape(-1, 1)
    cv = []
    for lam in lambdas:
        pipe = Pipeline([('scale_x', StandardScaler()),
                         ('poly', PolynomialFeatures(degree=10, include_bias=False)),
                         ('scale_f', StandardScaler()),
                         ('reg', Ridge(alpha=lam))])
        cv.append(-cross_val_score(pipe, X_an, F_an, cv=kfold, scoring='neg_mean_squared_error').mean())
    cv = np.array(cv); k = np.argmin(cv)
    flat = lambdas[cv < 1.1 * cv[k]]
    print(f"noise {noise:.2f} N: lambda* = {lambdas[k]:.3f}, min CV MSE = {cv[k]:.4f} N^2 "
          f"(sigma^2 = {noise**2:.4f}); within 10 % of the minimum: {flat.min():.3f}-{flat.max():.3f}")
    ax.loglog(lambdas, cv, 'o-', label=f'noise {noise} N')
ax.set_xlabel(r'$\lambda$'); ax.set_ylabel('5-fold CV MSE'); ax.legend()
fig.tight_layout(); fig.savefig('outputs/E5_4_lambda_vs_noise.png', dpi=150)
print("saved outputs/E5_4_lambda_vs_noise.png")
