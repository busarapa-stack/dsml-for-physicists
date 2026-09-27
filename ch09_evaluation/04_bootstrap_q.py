"""Ch.9  Pairs bootstrap of the damped-oscillator fit (code:bootstrap) and the Q table (tab:boot-q):
linear propagation, pairs bootstrap and parametric bootstrap for good (200 pts, 0.05 m) and poor (60 pts, 0.4 m) data.

Book references: ssec:nonparam-boot, ssec:param-boot, code:bootstrap (line for line), tab:boot-q
The listing continues from code:damped-cf of Chapter 5, which is executed first.
"""
import os
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
warnings.filterwarnings('ignore')
_s = (HERE.parent / 'ch05_regression' / '04_damped_oscillator.py').read_text()
exec(_s[_s.index('\n# --- code:damped-cf'):_s.index('\n# --- numbers quoted')])    # popt, pcov, t, x_obs, bounds

# --- code:bootstrap ----------------------------------------------------------
# continues from the damped-oscillator fit (popt, pcov, t, x_obs, bounds)
rng = np.random.default_rng(seed=1)
n_boot = 1000
params_boot = []
for _ in range(n_boot):
    idx = rng.integers(0, len(t), len(t))          # resample pairs (t_i, x_i)
    try:
        p_b, _ = curve_fit(damped_oscillator, t[idx], x_obs[idx],
                           p0=popt, bounds=bounds)
        params_boot.append(p_b)
    except RuntimeError:                           # rare non-convergence
        pass
params_boot = np.array(params_boot)
Q_boot = params_boot[:, 2] / (2 * params_boot[:, 1])
print("SE(gamma), SE(omega):", params_boot[:, 1:3].std(axis=0).round(4))
print("Q 95% percentile interval:", np.percentile(Q_boot, [2.5, 97.5]).round(2))
# --- end of listings ---------------------------------------------------------


def q_intervals(n, sig, n_boot=1000):
    rng_d = np.random.default_rng(seed=2026)
    tt = np.linspace(0, 10, n)
    xx = damped_oscillator(tt, 1.0, 0.3, 6.0, 0.5) + rng_d.normal(scale=sig, size=n)
    p, c = curve_fit(damped_oscillator, tt, xx, p0=[1.0, 0.2, 6.0, 0.0], bounds=bounds,
                     sigma=sig * np.ones(n), absolute_sigma=True)
    Q = p[2] / (2 * p[1])
    J = np.array([0, -p[2] / (2 * p[1]**2), 1 / (2 * p[1]), 0])
    sQ = np.sqrt(J @ c @ J)
    r = np.random.default_rng(1); pairs = []
    for _ in range(n_boot):
        i = r.integers(0, n, n)
        try:
            pairs.append(curve_fit(damped_oscillator, tt[i], xx[i], p0=p, bounds=bounds)[0])
        except RuntimeError:
            pass
    r = np.random.default_rng(1); param = []
    for _ in range(n_boot):
        xs = damped_oscillator(tt, *p) + r.normal(scale=sig, size=n)
        try:
            param.append(curve_fit(damped_oscillator, tt, xs, p0=p, bounds=bounds,
                                   sigma=sig * np.ones(n), absolute_sigma=True)[0])
        except RuntimeError:
            pass
    pairs, param = np.array(pairs), np.array(param)
    return dict(Q=Q, gamma=(p[1], np.sqrt(c[1, 1])), linear=(Q - 1.96 * sQ, Q + 1.96 * sQ),
                pairs=np.percentile(pairs[:, 2] / (2 * pairs[:, 1]), [2.5, 97.5]),
                param=np.percentile(param[:, 2] / (2 * param[:, 1]), [2.5, 97.5]), n_ok=(len(pairs), len(param)))


if __name__ == '__main__':
    print("curve_fit SE(gamma), SE(omega):", np.sqrt(np.diag(pcov))[1:3].round(4))
    print("\ntab:boot-q (true Q = 10):")
    for n, sig in [(200, 0.05), (60, 0.4)]:
        r = q_intervals(n, sig)
        print(f"  {n:3d} points, sigma {sig}: best Q {r['Q']:.2f}, gamma {r['gamma'][0]:.3f} +/- {r['gamma'][1]:.3f}; "
              f"linear [{r['linear'][0]:.2f}, {r['linear'][1]:.2f}]; pairs [{r['pairs'][0]:.2f}, {r['pairs'][1]:.2f}]; "
              f"parametric [{r['param'][0]:.2f}, {r['param'][1]:.2f}]  (fits converged {r['n_ok']})")
