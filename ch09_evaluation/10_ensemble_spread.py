"""Ch.9  Ensemble spread inside and outside the training range: bootstrap polynomials vs random forest.

Book references: ssec:ensemble, code:ensemble-spread (line for line)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.chdir(Path(__file__).resolve().parent)
Path('outputs').mkdir(exist_ok=True)

# --- code:ensemble-spread ----------------------------------------------------
from sklearn.ensemble import RandomForestRegressor

rng = np.random.default_rng(seed=2026)
x = rng.uniform(0, 3, 150)                       # training data only in [0, 3]
y = np.sin(2 * x) + rng.normal(scale=0.1, size=x.size)
x_grid = np.array([0.5, 1.5, 2.5, 3.5, 4.0])

poly = np.array([np.polyval(np.polyfit(x[i], y[i], 5), x_grid)
                 for i in (rng.integers(0, x.size, x.size) for _ in range(200))])
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                           random_state=0).fit(x[:, None], y)
trees = np.array([tree.predict(x_grid[:, None]) for tree in rf.estimators_])

print("truth      ", np.sin(2 * x_grid).round(2))
print("poly mean  ", poly.mean(0).round(2), " spread", poly.std(0).round(3))
print("forest mean", trees.mean(0).round(2), " spread", trees.std(0).round(3))
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    err_rf = np.abs(trees.mean(0) - np.sin(2 * x_grid))
    print(f"forest error at x = 3.5, 4.0: {err_rf[3:].round(2)}; forest spread inside [0, 3]: "
          f"{trees.std(0)[:3].min():.3f}-{trees.std(0)[:3].max():.3f}")
    xs = np.linspace(0, 4.2, 300)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, np.sin(2 * xs), 'k', label='truth'); ax.plot(x, y, '.', ms=3, alpha=0.4, label='training data')
    ax.plot(xs, rf.predict(xs[:, None]), label='random forest')
    ax.axvspan(3, 4.2, color='grey', alpha=0.15); ax.set_ylim(-2, 2); ax.legend()
    fig.tight_layout(); fig.savefig('outputs/ch09_ensemble_extrapolation.png', dpi=100)
    print("saved outputs/ch09_ensemble_extrapolation.png")
