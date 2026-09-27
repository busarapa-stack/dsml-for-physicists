"""Ch.5  Bias-variance trade-off: polynomial degree scan on an anharmonic spring.

Book references: ssec:bias-variance, code:bv-scan (reproduced line for line)
SYNTHETIC data: F = k1 x + k3 x^3 + noise, k1 = 5 N/m, k3 = 500 N/m^3, sigma = 0.05 N, 60 points.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

os.chdir(Path(__file__).resolve().parent)
Path('data').mkdir(exist_ok=True); Path('outputs').mkdir(exist_ok=True)

# --- code:bv-scan ------------------------------------------------------------
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(seed=2026)
k1, k3 = 5.0, 500.0                                  # N/m, N/m^3
x_an = rng.uniform(-0.15, 0.15, 60)
F_an = k1 * x_an + k3 * x_an**3 + rng.normal(scale=0.05, size=x_an.size)
X_an = x_an.reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X_an, F_an, test_size=0.3, random_state=42)
mse_train, mse_val = [], []
for d in range(1, 16):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('poly',   PolynomialFeatures(degree=d, include_bias=False)),
        ('reg',    LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    mse_train.append(np.mean((pipe.predict(X_train) - y_train)**2))
    mse_val.append(np.mean((pipe.predict(X_val) - y_val)**2))

# --- report ------------------------------------------------------------------
pd.DataFrame({'x': x_an, 'F': F_an}).to_csv('data/anharmonic_spring.csv', index=False)
print(" d   MSE_train    MSE_val   (noise variance sigma^2 = 0.0025 N^2)")
for d, a, b in zip(range(1, 16), mse_train, mse_val):
    print(f"{d:2d}   {a:9.5f}  {b:9.5f}")
best = int(np.argmin(mse_val)) + 1
print(f"lowest validation MSE at degree {best}")
print(f"degrees 1-2: MSE between {min(mse_train[:2] + mse_val[:2]):.3f} and {max(mse_train[:2] + mse_val[:2]):.3f} N^2 "
      f"= {min(mse_train[:2] + mse_val[:2]) / 0.0025:.0f}-{max(mse_train[:2] + mse_val[:2]) / 0.0025:.0f} x sigma^2")

# degree-3 polynomial on ALL data, raw x (coefficients keep their units)
fit3 = sm.OLS(F_an, np.column_stack([np.ones_like(x_an), x_an, x_an**2, x_an**3])).fit()
th, se = fit3.params, fit3.bse
print(f"degree 3 on all data: theta1 = {th[1]:.2f} +/- {se[1]:.2f} N/m, "
      f"theta2 = {th[2]:.1f} +/- {se[2]:.1f} N/m^2, theta3 = {th[3]:.0f} +/- {se[3]:.0f} N/m^3")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(range(1, 16), mse_train, 'o-', label='training')
ax.semilogy(range(1, 16), mse_val, 's-', label='validation')
ax.axhline(0.05**2, ls=':', color='k', label=r'$\sigma^2$')
ax.set_xlabel('polynomial degree'); ax.set_ylabel(r'MSE (N$^2$)'); ax.legend()
fig.tight_layout(); fig.savefig('outputs/ch05_bias_variance.png', dpi=150)
print("saved outputs/ch05_bias_variance.png")
