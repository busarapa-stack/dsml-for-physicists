"""E5.2  Slope from raw x and from standardized x: k_std = k_raw * s_x."""
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

os.chdir(Path(__file__).resolve().parent.parent)

rng = np.random.default_rng(seed=2026)                      # data of code:hooke
x = np.linspace(0.0, 0.20, 50)
F_obs = 5.0 * x + rng.normal(scale=0.03, size=x.size)
X = x.reshape(-1, 1)

k_raw = LinearRegression().fit(X, F_obs).coef_[0]
Z = StandardScaler().fit_transform(X)
k_std = LinearRegression().fit(Z, F_obs).coef_[0]
s_x = x.std()                                               # ddof = 0, as StandardScaler uses
print(f"raw slope          k_raw = {k_raw:.4f} N/m")
print(f"standardized slope k_std = {k_std:.4f} N   (units of F: z has no unit)")
print(f"k_raw * s_x (ddof=0)     = {k_raw * s_x:.4f} N   -> identical")
print(f"k_raw * s_x (ddof=1)     = {k_raw * x.std(ddof=1):.4f} N   -> differs slightly: StandardScaler divides by N")

# why scaling matters for a penalty: the same variable in metres and in millimetres
from sklearn.linear_model import Ridge
for unit, f in (('m', 1.0), ('mm', 1000.0)):
    c = Ridge(alpha=1.0).fit(X * f, F_obs).coef_[0] * f
    print(f"ridge (alpha = 1) with x in {unit:2s}: slope converted back to N/m = {c:.3f}")
print("-> the penalty depends on the unit of x unless the features are standardized first")
