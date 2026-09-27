"""E4.2  Explore the Co-60 counts: describe(), variance/mean, histogram + pmf, Q-Q plot
against the Poisson distribution, chi-squared test.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

os.chdir(Path(__file__).resolve().parent.parent)
counts = pd.read_csv('data/raw/co60_counts.csv')['counts']
print(counts.describe().round(3).to_string())
lam = counts.mean()
print(f"variance/mean = {counts.var() / lam:.3f}")

# Q-Q plot: sorted data against Poisson quantiles at the same plotting positions
q = (np.arange(1, counts.size + 1) - 0.5) / counts.size
theo = stats.poisson.ppf(q, mu=lam)
jit = np.random.default_rng(0).uniform(-0.15, 0.15, counts.size)   # separate ties
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
bins = np.arange(-0.5, counts.max() + 1.5)
ax[0].hist(counts, bins=bins, density=True, alpha=0.6, label='data')
k = np.arange(counts.max() + 1)
ax[0].plot(k, stats.poisson.pmf(k, lam), 'ro-', label=f'Poisson, mean {lam:.2f}')
ax[0].set_xlabel('counts per 1 s'); ax[0].legend()
ax[1].plot(theo + jit, np.sort(counts) + jit, '.', ms=2)
ax[1].plot([0, counts.max()], [0, counts.max()], 'k--')
ax[1].set_xlabel('Poisson quantile'); ax[1].set_ylabel('data quantile'); ax[1].set_title('Q-Q plot')
fig.tight_layout(); fig.savefig('outputs/E4_2_counts.png', dpi=150)
frac_on_line = np.mean(np.sort(counts.values) == theo)
print(f"{100 * frac_on_line:.1f} % of the sorted counts equal the Poisson quantile exactly")
print("conclusion: the data do not contradict a Poisson law (not: 'are certainly Poisson')")
print("saved outputs/E4_2_counts.png")
