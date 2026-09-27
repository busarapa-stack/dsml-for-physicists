"""Ch.4  Confidence intervals: Poisson mean, Wilson interval for a fraction, what '95 %'
means (coverage), and why small samples need Student's t.

Book references: ssec:ci, def:ci, code:wilson, ssec:ci-basics, tab:t-vs-z, review question 7
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

os.chdir(Path(__file__).resolve().parent)
counts = pd.read_csv('data/raw/co60_counts.csv')['counts'].values
n_bar, N = counts.mean(), counts.size
se = np.sqrt(n_bar / N)
print(f"Poisson mean: {n_bar:.3f} +/- {se:.3f}, 95 % CI [{n_bar - 1.96 * se:.3f}, {n_bar + 1.96 * se:.3f}]")

# --- code:wilson ------------------------------------------------------------
from statsmodels.stats.proportion import proportion_confint

n_pass, n_total = 3, 40       # events passing a selection cut
ci_lo, ci_hi = proportion_confint(n_pass, n_total, alpha=0.05, method='wilson')
print(f"fraction = {n_pass/n_total:.3f}, 95% CI = [{ci_lo:.3f}, {ci_hi:.3f}]")
f = n_pass / n_total
half = 1.96 * np.sqrt(f * (1 - f) / n_total)
print(f"Gaussian (Wald) interval f +/- 1.96 sqrt(f(1-f)/n): [{f - half:.3f}, {f + half:.3f}]"
      "  <- lower edge below 0, impossible for a fraction")

# --- def:ci : 95 % is a property of the METHOD (ring-toss analogy) ---------------
rng = np.random.default_rng(11)
true_p, hits_w, hits_n = 3 / 40, 0, 0
for _ in range(20000):
    k = rng.binomial(40, true_p)
    lo, hi = proportion_confint(k, 40, alpha=0.05, method='wilson'); hits_w += lo <= true_p <= hi
    lo, hi = proportion_confint(k, 40, alpha=0.05, method='normal'); hits_n += lo <= true_p <= hi
print(f"coverage over 20,000 repeated experiments (true fraction 0.075, n = 40): "
      f"Wilson {hits_w / 20000:.3f}, Wald {hits_n / 20000:.3f}")

# --- tab:t-vs-z ------------------------------------------------------------------
print("\n  N   t_0.025,N-1   wider than z = 1.96 by")
for n in (5, 10, 30, 100):
    t = stats.t.ppf(0.975, df=n - 1)
    print(f"{n:4d}   {t:8.3f}      {100 * (t / 1.96 - 1):5.1f} %")

# coverage of mean +/- 1.96 s/sqrt(N) versus mean +/- t s/sqrt(N) for N = 5
hits_z = hits_t = 0
for _ in range(20000):
    x = rng.normal(0, 1, 5)
    h = x.std(ddof=1) / np.sqrt(5)
    hits_z += abs(x.mean()) < 1.96 * h
    hits_t += abs(x.mean()) < stats.t.ppf(0.975, 4) * h
print(f"N = 5: '95 %' interval with 1.96 covers {hits_z / 20000:.3f}, with t covers {hits_t / 20000:.3f}")

# --- review question 7: pendulum, five measurements ---------------------------------
mean, s, n = 2.013, 0.021, 5
sem = s / np.sqrt(n)
t = stats.t.ppf(0.975, n - 1)
print(f"\npendulum: SEM = {sem:.4f} s; 95 % CI = {mean} +/- {t * sem:.3f} s "
      f"(with 1.96: +/- {1.96 * sem:.3f} s, {100 * (1 - 1.96 / t):.0f} % too narrow)")
print(f"halving a CI needs {(2)**2}x the data (SE ~ 1/sqrt(N))")
