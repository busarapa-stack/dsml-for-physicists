"""Ch.9  Bayesian fit of the damped oscillator with PyMC (NUTS), posterior predictive checks,
and PSIS-LOO / WAIC comparison of M1, M2, M3.

Book references: ssec:mcmc, code:pymc (line for line), ssec:ppc, ssec:waic-loo.  Needs pymc and arviz.
Continues from code:damped-cf (Chapter 5).  The first run compiles PyTensor code (can take minutes);
after that about 1 minute (four models).
"""
import logging
import os
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
warnings.filterwarnings('ignore')
_s = (HERE.parent / 'ch05_regression' / '04_damped_oscillator.py').read_text()
exec(_s[_s.index('\n# --- code:damped-cf'):_s.index('\n# --- numbers quoted')])      # t, x_obs

# --- code:pymc ---------------------------------------------------------------
import pymc as pm
import arviz as az

with pm.Model() as model:
    A     = pm.HalfNormal('A', sigma=2.0)                  # A > 0
    gamma = pm.HalfNormal('gamma', sigma=1.0)              # gamma >= 0
    omega = pm.TruncatedNormal('omega', mu=6.0, sigma=1.0, lower=0.0)  # from k and m
    phi   = pm.Uniform('phi', lower=-np.pi, upper=np.pi)
    sigma = pm.HalfNormal('sigma', sigma=0.2)              # noise level, unknown
    mu = A * pm.math.exp(-gamma * t) * pm.math.cos(omega * t + phi)
    pm.Normal('x', mu=mu, sigma=sigma, observed=x_obs)
    idata = pm.sample(2000, tune=1000, chains=4, random_seed=2026,
                      idata_kwargs={'log_likelihood': True})
    pm.sample_posterior_predictive(idata, extend_inferencedata=True,
                                   random_seed=2026)
print(az.summary(idata, var_names=['A', 'gamma', 'omega', 'phi', 'sigma'],
                 hdi_prob=0.95))
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    logging.getLogger('pymc').setLevel(logging.ERROR)
    post = idata.posterior.stack(s=('chain', 'draw'))
    A_, g_, w_, p_, s_ = (post[v].values for v in ('A', 'gamma', 'omega', 'phi', 'sigma'))
    print(f"\ngamma posterior sd {g_.std():.4f}; ess_bulk "
          f"{az.summary(idata, var_names=['A', 'gamma', 'omega', 'phi', 'sigma'])['ess_bulk'].agg(['min', 'max']).tolist()}")
    mu = A_[:, None] * np.exp(-g_[:, None] * t) * np.cos(w_[:, None] * t + p_[:, None])
    yrep = idata.posterior_predictive['x'].stack(s=('chain', 'draw')).values.T

    def ppc_stats(y):
        r = (y - mu) / s_[:, None]
        return (r**2).sum(1), np.abs(r).max(1), np.array([np.corrcoef(ri[:-1], ri[1:])[0, 1] for ri in r])

    obs, rep = ppc_stats(np.broadcast_to(x_obs, mu.shape)), ppc_stats(yrep)
    for name, o, r in zip(['sum (resid/sigma)^2', 'max |resid/sigma|', 'lag-1 resid correlation'], obs, rep):
        print(f"posterior predictive p-value, {name:24s}: {(r >= o).mean():.2f}")

    def fit_model(kind):
        with pm.Model():
            A = pm.HalfNormal('A', sigma=2.0)
            omega = pm.TruncatedNormal('omega', mu=6.0, sigma=1.0, lower=0.0)
            phi = pm.Uniform('phi', lower=-np.pi, upper=np.pi)
            sigma = pm.HalfNormal('sigma', sigma=0.2)
            if kind == 'M1':
                mu_ = A * pm.math.cos(omega * t + phi)
            else:
                gamma = pm.HalfNormal('gamma', sigma=1.0)
                mu_ = A * pm.math.exp(-gamma * t) * pm.math.cos(omega * t + phi)
                if kind == 'M3':
                    mu_ = mu_ + pm.Normal('c', mu=0.0, sigma=0.5)       # zero offset, weak prior
            pm.Normal('x', mu=mu_, sigma=sigma, observed=x_obs)
            return pm.sample(2000, tune=1000, chains=4, random_seed=2026, progressbar=False,
                             idata_kwargs={'log_likelihood': True})

    ids = {'M1': fit_model('M1'), 'M2': idata, 'M3': fit_model('M3')}
    for ic in ('loo', 'waic'):
        print(f"\naz.compare, {ic}:")
        print(az.compare(ids, ic=ic).iloc[:, :7].round(2).to_string())
    print("max Pareto k:", {k: round(float(az.loo(v, pointwise=True).pareto_k.max()), 2) for k, v in ids.items()})
