"""Ch.4  Are Co-60 counts Poisson? variance/mean, histogram with the pmf, chi-squared test.

Book references: ssec:poisson-fit, ex:poisson-fit, code:poisson-fit
Counts are SYNTHETIC: rng.poisson(lam=5, size=1000), the recipe given in E4.2.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
os.chdir(Path(__file__).resolve().parent)

# --- code:poisson-fit -------------------------------------------------------
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

counts = pd.read_csv('data/raw/co60_counts.csv')['counts'].values
lam_hat = counts.mean()
print(f"variance/mean = {counts.var(ddof=1)/lam_hat:.3f}")   # near 1?

# histogram with the Poisson pmf on top
n_max = counts.max() + 1
bins = np.arange(-0.5, n_max + 0.5, 1)
fig, ax = plt.subplots()
ax.hist(counts, bins=bins, density=True, alpha=0.6, label='data')
n_range = np.arange(0, n_max)
ax.plot(n_range, stats.poisson.pmf(n_range, mu=lam_hat), 'ro-', label='Poisson')
ax.set_xlabel('counts per 1 s'); ax.set_ylabel('probability'); ax.legend()

# chi-squared goodness of fit, keep bins with expected count >= 5
observed, _ = np.histogram(counts, bins=bins)
expected = len(counts) * stats.poisson.pmf(n_range, mu=lam_hat)
mask = expected >= 5
obs, exp = observed[mask], expected[mask]
exp = exp * obs.sum() / exp.sum()          # same total in both
chi2_stat, p = stats.chisquare(obs, f_exp=exp, ddof=1)   # 1 fitted parameter
print(f"chi^2 = {chi2_stat:.2f}, dof = {mask.sum() - 2}, p = {p:.3f}")

fig.savefig('outputs/ch04_poisson_fit.png', dpi=150)

# --- extra checks ---------------------------------------------------------------
print(f"\nmean = {lam_hat:.3f} counts/s from {counts.size} intervals; "
      f"kept {mask.sum()} of {n_range.size} bins (expected >= 5)")
# how much can variance/mean fluctuate for a true Poisson sample of this size?
rng = np.random.default_rng(1)
ratios = [ (x := rng.poisson(5, 1000)).var(ddof=1) / x.mean() for _ in range(2000)]
lo, hi = np.percentile(ratios, [2.5, 97.5])
print(f"for true Poisson data (N = 1000) variance/mean falls in [{lo:.3f}, {hi:.3f}] 95 % of the time")
print("saved outputs/ch04_poisson_fit.png")
