"""E5.3  Polynomials of degree 1, 3, 6 and 10 fitted to the training part of code:bv-scan,
and how much each curve changes when the data are drawn again (variance)."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

os.chdir(Path(__file__).resolve().parent.parent)


def make(seed):
    rng = np.random.default_rng(seed=seed)
    x = rng.uniform(-0.15, 0.15, 60)
    F = 5.0 * x + 500.0 * x**3 + rng.normal(scale=0.05, size=x.size)
    return train_test_split(x.reshape(-1, 1), F, test_size=0.3, random_state=42)


def model(d):
    return Pipeline([('scaler', StandardScaler()),
                     ('poly', PolynomialFeatures(degree=d, include_bias=False)),
                     ('reg', LinearRegression())])


X_train, X_val, y_train, y_val = make(2026)
xg = np.linspace(-0.15, 0.15, 400).reshape(-1, 1)
truth = 5.0 * xg[:, 0] + 500.0 * xg[:, 0]**3
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(X_train, y_train, 'k.', label='training data')
ax.plot(xg, truth, 'k:', lw=1, label='true law')
for d in (1, 3, 6, 10):
    m = model(d).fit(X_train, y_train)
    ax.plot(xg, m.predict(xg), label=f'degree {d}')
    r = y_train - m.predict(X_train)
    # residual sign runs: a systematic pattern gives few long runs
    runs = 1 + np.sum(np.diff(np.sign(r[np.argsort(X_train[:, 0])])) != 0)
    print(f"degree {d:2d}: largest |curve - true law| on the grid {np.max(np.abs(m.predict(xg) - truth)):.3f} N, "
          f"residual sign runs {runs} of {len(r)} points")
ax.set_ylim(-3, 3); ax.set_xlabel('x (m)'); ax.set_ylabel('F (N)'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig('outputs/E5_3_degrees.png', dpi=150)

# variance: redraw the data 200 times and look at the spread of each curve
print("\nspread (std) of the fitted curve over 200 new data sets, at the centre and near the edge:")
edge = np.array([[0.145]]); centre = np.array([[0.0]])
for d in (1, 3, 6, 10):
    pe, pc = [], []
    for s in range(200):
        Xt, _, yt, _ = make(10_000 + s)
        m = model(d).fit(Xt, yt)
        pe.append(m.predict(edge)[0]); pc.append(m.predict(centre)[0])
    print(f"degree {d:2d}: x = 0: {np.std(pc):.3f} N   x = 0.145 m: {np.std(pe):.3f} N")
print("saved outputs/E5_3_degrees.png")
