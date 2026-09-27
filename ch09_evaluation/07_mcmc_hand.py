"""Ch.9  Hand-written Metropolis sampler for the damped-oscillator posterior (uniform prior inside the bounds, known sigma).

Book references: ssec:mcmc, code:mcmc-hand (line for line)
Continues from code:damped-cf (Chapter 5) and uses tau_int from code:mc-tau (05_mc_tau.py, function definitions only).
"""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
_s = (HERE.parent / 'ch05_regression' / '04_damped_oscillator.py').read_text()
exec(_s[_s.index('\n# --- code:damped-cf'):_s.index('\n# --- numbers quoted')])      # popt, pcov, t, x_obs, sigma_x, bounds
_s = (HERE / '05_mc_tau.py').read_text()
exec(_s[_s.index('def tau_int'):_s.index('def block_bootstrap')])                      # tau_int only

# --- code:mcmc-hand ----------------------------------------------------------
# continues from the damped-oscillator fit (popt, pcov, t, x_obs, sigma_x, bounds)
lo, hi = np.array(bounds[0]), np.array(bounds[1])

def log_posterior(theta):
    if np.any(theta < lo) or np.any(theta > hi):      # uniform prior inside bounds
        return -np.inf
    chi2 = np.sum(((x_obs - damped_oscillator(t, *theta)) / sigma_x)**2)
    return -0.5 * chi2                                 # ln L + const

rng = np.random.default_rng(seed=1)
step = 2.38**2 / 4 * pcov                              # proposal covariance
theta, lp = popt.copy(), log_posterior(popt)
chain, n_acc = [], 0
for k in range(50000):
    prop = rng.multivariate_normal(theta, step)
    lp_prop = log_posterior(prop)
    if np.log(rng.random()) < lp_prop - lp:            # Metropolis rule, T = 1
        theta, lp = prop, lp_prop
        n_acc += 1
    chain.append(theta)
chain = np.array(chain)[5000:]                         # drop burn-in
print("acceptance:", n_acc / 50000)
for name, c in zip(['A', 'gamma', 'omega', 'phi'], chain.T):
    lo95, med, hi95 = np.percentile(c, [2.5, 50, 97.5])
    print(f"{name:5s} median {med:.4f}  95% [{lo95:.4f}, {hi95:.4f}]  "
          f"tau_int {tau_int(c)[0]:.1f}")
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    print(f"posterior sd: gamma {chain[:, 1].std():.4f}, omega {chain[:, 2].std():.4f}; curve_fit SE "
          f"{np.sqrt(pcov[1, 1]):.4f}, {np.sqrt(pcov[2, 2]):.4f}")
    taus = [tau_int(c)[0] for c in chain.T]
    print(f"independent samples ~ {len(chain) / (2 * np.mean(taus)):.0f} of {len(chain)} (1 in {2 * np.mean(taus):.0f})")
