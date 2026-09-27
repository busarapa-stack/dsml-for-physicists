"""Ch.9  Shuffled vs blocked cross-validation on time-correlated data, and a new independent run.

Book references: ssec:kfold-cv, code:blocked-cv (line for line)
SYNTHETIC AR(1) data (seed 2026; new run seed 7).
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

# --- code:blocked-cv ---------------------------------------------------------
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

def ar1(n, phi, sigma, rng):
    """AR(1) process: e[i] = phi * e[i-1] + white noise."""
    e = np.empty(n)
    e[0] = rng.normal(scale=sigma / np.sqrt(1 - phi**2))
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal(scale=sigma)
    return e

def make_run(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 1.0
    x = ar1(n, 0.995, 0.1, rng)                 # slowly varying control
    y = 2.0 * x + ar1(n, 0.99, 0.07, rng)       # response + correlated noise
    return np.c_[x, t], y

X, y = make_run(seed=2026)
X_new, y_new = make_run(seed=7)                 # an independent new run
model = RandomForestRegressor(200, min_samples_leaf=5, random_state=0, n_jobs=-1)
for cols in ([0], [0, 1]):                      # x only / x and t
    for name, cv in [('shuffled', KFold(5, shuffle=True, random_state=0)),
                     ('blocked',  KFold(5, shuffle=False))]:
        rmse = -cross_val_score(model, X[:, cols], y, cv=cv,
                                scoring='neg_root_mean_squared_error')
        print(cols, name, rmse.mean().round(3))
    model.fit(X[:, cols], y)
    rmse_new = np.sqrt(np.mean((model.predict(X_new[:, cols]) - y_new)**2))
    print(cols, 'new run', rmse_new.round(3))
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    noise_sd = 0.07 / np.sqrt(1 - 0.99**2)
    print(f"\nstationary sd of the correlated noise: {noise_sd:.3f} (floor for any model on a new run); "
          f"sample sd in this run {(y - 2 * X[:, 0]).std():.3f}")
