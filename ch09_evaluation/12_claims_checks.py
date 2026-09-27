"""Ch.9  Second-pass checks of statements in the text that have no listing of their own.

  1  all ten listings run in book order as one notebook, with the Chapter 4/5/6/8 code they continue from
  2  two 5-fold training sets overlap by 75 %                                     (ssec:cv-PI)
  3  ln N > 2 from N = 8                                                          (ssec:ic)
  4  poor-data Q: bootstrap SD is dominated by a few gamma ~ 0 resamples          (tab:boot-q text)
  5  blocks of 50 sweeps underestimate the SE of <e> by about a quarter           (ssec:boot-mc-primary)
  6  the balanced forest calls 'signal' more often than KNN at its default cut    (McNemar paragraph)
  7  AUC is unchanged when every score is cubed                                  (notebox, ssec:calibration)
  8  E9.4: posterior of Q with a log-uniform prior on gamma instead of uniform     (E9.4 answer)
  9  posterior predictive check of a WRONG model (M1, undamped): chi2 p-value vs lag-1 correlation
                                                                                 (notebox in ssec:ppc)
About 5 minutes (the Metropolis run of code:mc-tau dominates).
"""
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
warnings.filterwarnings('ignore')
logging.getLogger('pymc').setLevel(logging.ERROR)


def section(path, start, stop):
    s = Path(path).read_text()
    s = s[s.index(start):]
    return s[:s.index(stop)]


# 1 ------------------------------------------------------------------------------------------------
t0 = time.time()
nb = {'__name__': 'notebook'}
exec(section(HERE / '01_blocked_cv.py', '\n# --- code:blocked-cv', '\n# --- end of listings'), nb)
exec(section(HERE / '03_nested_cv.py', '\n# --- code:nested-cv', '\n# --- end of listings'), nb)
exec(section(HERE.parent / 'ch05_regression/04_damped_oscillator.py', '\n# --- code:damped-cf', '\n# --- numbers quoted'), nb)
exec(section(HERE / '04_bootstrap_q.py', '\n# --- code:bootstrap', '\n# --- end of listings'), nb)
exec(section(HERE.parent / 'ch04_statistics/07_metropolis_ising.py', '\n# --- code:metropolis', 'def simulate_T'), nb)
exec(section(HERE / '05_mc_tau.py', '\n# --- code:mc-tau', '\n# --- end of listings'), nb)
for f, lab in [('07_mcmc_hand.py', 'mcmc-hand'), ('08_pymc_oscillator.py', 'pymc')]:
    exec(section(HERE / f, f'\n# --- code:{lab}', '\n# --- end of listings'), nb)
_ch6 = HERE.parent / 'ch06_classification'
for f in ('01_higgs_toy_data.py', '02_logreg_forest.py', '03_compare_five.py'):
    nb['__file__'] = str(_ch6 / f)
    exec(section(_ch6 / f, '\n# --- code:', '\n# --- end of listings'), nb)
os.chdir(HERE)
exec(section(HERE / '09_delong_calibration.py', '\n# --- code:delong', '\n# --- end of listings'), nb)
exec(section(HERE / '10_ensemble_spread.py', '\n# --- code:ensemble-spread', '\n# --- end of listings'), nb)
exec(section(HERE.parent / 'ch08_dimred/04_rouse_pca.py', '\n# --- code:rouse-gen', '\n# --- code:polymer-pca'), nb)
exec(section(HERE / '11_bayes_cov_polymer.py', '\n# --- code:bayes-cov', '\n# --- end of listings'), nb)
print(f"\n1  all ten Chapter 9 listings ran in book order as one notebook ({time.time() - t0:.0f} s)")

# 2 ------------------------------------------------------------------------------------------------
print(f"2  5-fold CV: each training set holds 4/5 of the data; two of them share 3/5, i.e. {100 * (3 / 5) / (4 / 5):.0f} % of each")

# 3 ------------------------------------------------------------------------------------------------
print(f"3  ln 7 = {np.log(7):.2f}, ln 8 = {np.log(8):.2f}")

# 4 ------------------------------------------------------------------------------------------------
from scipy.optimize import curve_fit
rng_d = np.random.default_rng(seed=2026)
tt = np.linspace(0, 10, 60)
xx = nb['damped_oscillator'](tt, 1.0, 0.3, 6.0, 0.5) + rng_d.normal(scale=0.4, size=60)
p, c = curve_fit(nb['damped_oscillator'], tt, xx, p0=[1, .2, 6, 0], bounds=nb['bounds'], sigma=0.4 * np.ones(60), absolute_sigma=True)
r = np.random.default_rng(1); Qb = []
for _ in range(1000):
    i = r.integers(0, 60, 60)
    try:
        pb = curve_fit(nb['damped_oscillator'], tt[i], xx[i], p0=p, bounds=nb['bounds'])[0]
        Qb.append(pb[2] / (2 * pb[1]))
    except RuntimeError:
        pass
Qb = np.array(Qb)
print(f"4  poor data: gamma = {p[1]:.2f} +/- {np.sqrt(c[1, 1]):.2f} ({100 * np.sqrt(c[1, 1]) / p[1]:.0f} %); bootstrap Q: median {np.median(Qb):.1f}, "
      f"SD {Qb.std():.0f}, max {Qb.max():.0f}; SD without the top 1 % {np.sort(Qb)[:int(0.99 * len(Qb))].std():.1f}")

# 5 ------------------------------------------------------------------------------------------------
e = nb['e']
se = {b: nb['block_bootstrap'](e, np.mean, block=b).std() for b in (50, 200, 500)}
print(f"5  block SE of <e>: 50 -> {se[50]:.4f}, 200 -> {se[200]:.4f}, 500 -> {se[500]:.4f}; "
      f"blocks of 50 are {100 * (1 - se[50] / 0.0071):.0f} % below the plateau 0.0071")

# 6 ------------------------------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
calls = {}
for n in ('KNN', 'RandForest'):
    calls[n] = Pipeline([('scaler', StandardScaler()), ('clf', nb['classifiers'][n])]).fit(nb['X_train'], nb['y_train']).predict(nb['X_val'])
yv = nb['y_val']
print(f"6  fraction called signal: KNN {calls['KNN'].mean():.3f}, balanced RF {calls['RandForest'].mean():.3f} (true {yv.mean():.3f}); "
      f"signal efficiency KNN {calls['KNN'][yv == 1].mean():.3f}, RF {calls['RandForest'][yv == 1].mean():.3f}")

# 7 ------------------------------------------------------------------------------------------------
s = nb['score']['RandForest']
print(f"7  AUC of RF scores {roc_auc_score(yv, s):.4f}, of scores cubed {roc_auc_score(yv, s**3):.4f}; "
      f"mean score {s.mean():.3f} vs cubed {np.mean(s**3):.3f}")

# 8 ------------------------------------------------------------------------------------------------
lo_b, hi_b = np.array(nb['bounds'][0]), np.array(nb['bounds'][1])


def run_chain(log_prior, n=100000, seed=1):
    def lp(th):
        if np.any(th < lo_b) or np.any(th > hi_b) or th[1] <= 0:
            return -np.inf
        return -0.5 * np.sum(((xx - nb['damped_oscillator'](tt, *th)) / 0.4)**2) + log_prior(th)
    rg = np.random.default_rng(seed)
    step = 2.38**2 / 4 * c
    th, l = p.copy(), lp(p); ch = []
    for _ in range(n):
        pr = rg.multivariate_normal(th, step); lpr = lp(pr)
        if np.log(rg.random()) < lpr - l:
            th, l = pr, lpr
        ch.append(th)
    return np.array(ch)[10000:]


for name, prior in [('uniform in gamma', lambda th: 0.0), ('uniform in ln gamma', lambda th: -np.log(th[1]))]:
    ch = run_chain(prior)
    Q = ch[:, 2] / (2 * ch[:, 1])
    print(f"8  E9.4 with prior {name:20s}: gamma median {np.median(ch[:, 1]):.2f}, Q median {np.median(Q):.1f}, "
          f"95 % [{np.percentile(Q, 2.5):.1f}, {np.percentile(Q, 97.5):.1f}]")

# 9 ------------------------------------------------------------------------------------------------
import pymc as pm
t, x_obs = nb['t'], nb['x_obs']
with pm.Model():
    A = pm.HalfNormal('A', sigma=2.0)
    omega = pm.TruncatedNormal('omega', mu=6.0, sigma=1.0, lower=0.0)
    phi = pm.Uniform('phi', lower=-np.pi, upper=np.pi)
    sigma = pm.HalfNormal('sigma', sigma=0.2)
    pm.Normal('x', mu=A * pm.math.cos(omega * t + phi), sigma=sigma, observed=x_obs)
    id1 = pm.sample(1000, tune=1000, chains=4, random_seed=2026, progressbar=False)
    pm.sample_posterior_predictive(id1, extend_inferencedata=True, random_seed=2026, progressbar=False)
po = id1.posterior.stack(s=('chain', 'draw'))
mu = po['A'].values[:, None] * np.cos(po['omega'].values[:, None] * t + po['phi'].values[:, None])
sg = po['sigma'].values[:, None]
yrep = id1.posterior_predictive['x'].stack(s=('chain', 'draw')).values.T


def stats3(y):
    rr = (y - mu) / sg
    return (rr**2).sum(1), np.array([np.corrcoef(ri[:-1], ri[1:])[0, 1] for ri in rr])


so, sr = stats3(np.broadcast_to(x_obs, mu.shape)), stats3(yrep)
print(f"9  WRONG model M1 (no damping): sigma estimated {np.median(po['sigma'].values):.2f} m (true 0.05); "
      f"PPC p-value of chi2 {(sr[0] >= so[0]).mean():.2f}, of lag-1 residual correlation {(sr[1] >= so[1]).mean():.3f} "
      f"(observed lag-1 r = {so[1].mean():.2f})")
