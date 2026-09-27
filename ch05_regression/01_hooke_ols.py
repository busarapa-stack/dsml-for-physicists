"""Ch.5  Hooke's law by ordinary least squares: scikit-learn, statsmodels, and a pipeline.

Book references: ssec:sklearn-hooke, ssec:coeff-se, ssec:sklearn-pipeline, ssec:err-prop
Listings reproduced line for line: code:hooke, code:statsmodels, code:pipeline
All data are SYNTHETIC (k_true = 5.0 N/m, noise 0.03 N, seed 2026).
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

# --- code:hooke --------------------------------------------------------------
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(seed=2026)
k_true = 5.0                                   # N/m
x = np.linspace(0.0, 0.20, 50)                 # m
F_obs = k_true * x + rng.normal(scale=0.03, size=x.size)   # N

X = x.reshape(-1, 1)                           # shape (50, 1)
model = LinearRegression()
model.fit(X, F_obs)
print(f"k_fit = {model.coef_[0]:.4f} N/m, intercept = {model.intercept_:.4f} N")
print(f"R^2 = {model.score(X, F_obs):.4f}")

# --- code:statsmodels --------------------------------------------------------
import statsmodels.api as sm

X_ic = sm.add_constant(X)             # add the column of ones for the intercept
ols_model = sm.OLS(F_obs, X_ic).fit()
print(ols_model.summary())            # coef, SE, t, p, 95% CI
print(ols_model.bse)                  # standard errors only
print(ols_model.conf_int(alpha=0.05)) # 95% confidence intervals

# --- numbers quoted in the text ---------------------------------------------
k_hat, se_k = ols_model.params[1], ols_model.bse[1]
lo, hi = ols_model.conf_int(alpha=0.05)[1]
print(f"\nk = {k_hat:.3f} +/- {se_k:.3f} N/m, 95 % CI [{lo:.2f}, {hi:.2f}] N/m")
print(f"k_true = 5.0 lies {(k_hat - k_true) / se_k:.1f} SE from the estimate "
      f"(outside the 95 % CI: {not lo <= k_true <= hi}); relative error {100 * (k_hat / k_true - 1):.1f} %")
cov = ols_model.cov_params()
print(f"correlation(intercept, slope) = {cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]):.2f}")
print(f"sum of residuals = {ols_model.resid.sum():.1e} (zero when the model has an intercept)")

# how often does the 95 % CI miss k_true?  repeat the experiment with other seeds
miss = 0
for s in range(2000):
    r = np.random.default_rng(s)
    F = k_true * x + r.normal(scale=0.03, size=x.size)
    lo_s, hi_s = sm.OLS(F, X_ic).fit().conf_int(alpha=0.05)[1]
    miss += not lo_s <= k_true <= hi_s
print(f"2,000 repeated experiments: the 95 % CI misses k_true in {miss / 2000:.3f} of them")

# --- code:pipeline -----------------------------------------------------------
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
    ('reg',    LinearRegression()),
])
pipe.fit(X, F_obs)          # X must be 2-D, as in the previous listing
F_pred = pipe.predict(X)
print(f"\npipeline (scale -> degree 2 -> OLS): MSE = {np.mean((F_pred - F_obs)**2):.2e} N^2")
