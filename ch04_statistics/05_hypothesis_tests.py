"""Ch.4  Hypothesis tests on the Co-60 counts, and the numbers quoted in sec:inference.

Book references: ssec:hypothesis-testing, ssec:chi2-ttest, code:chi2-ttest,
                 look-elsewhere effect, review question 4
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

os.chdir(Path(__file__).resolve().parent)
# the Poisson example (code:poisson-fit) provides counts, obs and exp
counts = pd.read_csv('data/raw/co60_counts.csv')['counts'].values
lam_hat = counts.mean()
bins = np.arange(-0.5, counts.max() + 1.5, 1)
n_range = np.arange(0, counts.max() + 1)
observed, _ = np.histogram(counts, bins=bins)
expected = len(counts) * stats.poisson.pmf(n_range, mu=lam_hat)
mask = expected >= 5
obs, exp = observed[mask], expected[mask]
exp = exp * obs.sum() / exp.sum()

# --- code:chi2-ttest --------------------------------------------------------
# chi-squared goodness of fit (reuse obs, exp from the Poisson example)
chi2_stat, p = stats.chisquare(obs, f_exp=exp, ddof=1)
print("H0 (Poisson):", "reject" if p < 0.05 else "do not reject",
      f"(p = {p:.3f})")

# one-sample t-test against the expected rate 5.0 counts/s
t_stat, p_t = stats.ttest_1samp(counts, popmean=5.0)
print(f"t = {t_stat:.3f}, p = {p_t:.3f}")

# --- what the t-test result means here -----------------------------------------
print(f"\nThese counts were GENERATED from a Poisson law with mean exactly 5.0, yet the"
      f" t-test gives p = {p_t:.3f}.")
print("With alpha = 0.05, true-H0 data are rejected 5 % of the time (type I error);"
      " this sample is one of those. One test on one data set is not a discovery.")
rng = np.random.default_rng(7)
p_sim = np.array([stats.ttest_1samp(rng.poisson(5, 1000), 5.0).pvalue for _ in range(4000)])
print(f"check with 4,000 simulated experiments: fraction with p < 0.05 = {np.mean(p_sim < 0.05):.3f}")

# --- numbers quoted in the section ------------------------------------------------
print(f"\n5 sigma, one-sided: alpha = {stats.norm.sf(5):.2e}  (text: ~3e-7)")
print(f"20 independent tests at 0.05: P(at least one false positive) = {1 - 0.95**20:.2f}  (text: ~64 %)")
print(f"50 mass windows at local p < 0.01: P(at least one) = {1 - 0.99**50:.2f}; "
      f"Bonferroni threshold 0.05/50 = {0.05 / 50}")
