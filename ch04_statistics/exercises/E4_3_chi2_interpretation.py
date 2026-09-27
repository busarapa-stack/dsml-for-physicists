"""E4.3  What a small or a large p-value can mean physically: simulate counts with a
dead-time counter (variance < mean) and with a drifting background (variance > mean),
and ask how far from Poisson the real data could be (CI of variance/mean).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

os.chdir(Path(__file__).resolve().parent.parent)
rng = np.random.default_rng(43)


def chi2_poisson(counts):
    lam = counts.mean()
    bins = np.arange(-0.5, counts.max() + 1.5)
    k = np.arange(counts.max() + 1)
    obs, _ = np.histogram(counts, bins)
    exp = len(counts) * stats.poisson.pmf(k, lam)
    m = exp >= 5
    o, e = obs[m], exp[m] * obs[m].sum() / exp[m].sum()
    return stats.chisquare(o, e, ddof=1).pvalue


def dead_time_counts(rate, tau, n, T=1.0):
    """Paralyzable-free (non-paralyzable) counter: events closer than tau are lost."""
    out = []
    for _ in range(n):
        t = np.sort(rng.uniform(0, T, rng.poisson(rate * T)))
        kept, last = 0, -np.inf
        for ti in t:
            if ti - last >= tau:
                kept += 1; last = ti
        out.append(kept)
    return np.array(out)


data = pd.read_csv('data/raw/co60_counts.csv')['counts'].values
cases = {
    'Co-60 data (this chapter)': data,
    'dead time: rate 12/s, tau 0.05 s': dead_time_counts(12, 0.05, 1000),
    'background drifting 3 -> 7 /s': rng.poisson(np.linspace(3, 7, 1000)),
}
print(f"{'case':36s} {'mean':>6s} {'var/mean':>9s} {'chi2 p':>8s}")
for name, c in cases.items():
    print(f"{name:36s} {c.mean():6.2f} {c.var(ddof=1) / c.mean():9.3f} {chi2_poisson(c):8.3g}")

# (c) a high p does not prove Poisson: 95 % CI of variance/mean by bootstrap
boot = []
for _ in range(4000):
    s = rng.choice(data, data.size)
    boot.append(s.var(ddof=1) / s.mean())
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\nCo-60 data: variance/mean 95 % bootstrap CI [{lo:.3f}, {hi:.3f}] -> deviations from"
      " Poisson of up to ~8 % either way are not excluded")
