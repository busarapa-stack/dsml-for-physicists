"""Ch.10  Physics-informed neural network (PINN) for the damped oscillator vs a data-only network vs curve_fit.

Book references: ssec:pinn-example, code:pinn (line for line); damped_oscillator from Chapter 5 (code:damped-cf)
Numbers quoted in the text: both trainings about two minutes; the network has ~8,500 parameters;
data only: RMSE 0.39 m (t <= 3) and 0.34 m (t > 3); PINN: 0.036 m and 0.111 m, gamma = 0.327, omega_0 = 6.020;
PINN error grows with distance from the data: 0.080 m for 3-4.5 s, 0.135 m for 4.5-6 s, where its amplitude
is ~0.07 m against 0.25 m (drifting toward the trivial solution x = 0);
curve_fit of A exp(-gamma t) cos(omega t + phi) on the same 30 points: gamma = 0.264 +/- 0.038,
omega = 6.038 +/- 0.037, RMSE 0.026 m and 0.031 m, in well under a second.
About 2 minutes.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import ch5_damped

os.chdir(HERE)
Path('outputs').mkdir(exist_ok=True)
damped_oscillator = ch5_damped({'__name__': 'ch05'})['damped_oscillator']
_t0 = time.time()

# --- code:pinn ---------------------------------------------------------------
import torch
import torch.nn as nn

rng = np.random.default_rng(seed=2026)
t_obs = np.sort(rng.uniform(0, 3, 30))                 # data only in the first 3 s
x_obs = damped_oscillator(t_obs, 1.0, 0.3, 6.0, 0.5) + rng.normal(scale=0.05, size=30)
t_d = torch.tensor(t_obs, dtype=torch.float32)[:, None]
x_d = torch.tensor(x_obs, dtype=torch.float32)[:, None]
t_c = torch.linspace(0, 6, 300)[:, None].requires_grad_(True)   # collocation points

def train(use_physics, n_iter=30000, lam=1e-3, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                        nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    log_g = torch.tensor(np.log(0.5), requires_grad=True)     # unknown gamma
    log_w2 = torch.tensor(np.log(30.0), requires_grad=True)   # unknown omega_0^2
    params = list(net.parameters()) + ([log_g, log_w2] if use_physics else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    for it in range(n_iter):
        loss = torch.mean((net(t_d) - x_d)**2)                 # data term
        if use_physics:                                        # ODE residual term
            x = net(t_c)
            dx = torch.autograd.grad(x.sum(), t_c, create_graph=True)[0]
            ddx = torch.autograd.grad(dx.sum(), t_c, create_graph=True)[0]
            res = ddx + 2 * torch.exp(log_g) * dx + torch.exp(log_w2) * x
            loss = loss + lam * torch.mean(res**2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net, np.exp(log_g.item()), np.sqrt(np.exp(log_w2.item()))

t_test = np.linspace(0, 6, 601)
x_true = damped_oscillator(t_test, 1.0, 0.3, 6.0, 0.5)
for use_physics in (False, True):
    net, g, w0 = train(use_physics)
    with torch.no_grad():
        x_pred = net(torch.tensor(t_test, dtype=torch.float32)[:, None]).numpy().ravel()
    err = x_pred - x_true
    print(f"physics={use_physics}: RMSE t<=3 {np.sqrt(np.mean(err[t_test<=3]**2)):.3f}, "
          f"t>3 {np.sqrt(np.mean(err[t_test>3]**2)):.3f}")
    if use_physics:
        print(f"  gamma = {g:.3f} (true 0.3), omega_0 = {w0:.3f} (true 6.007)")
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    print(f"both trainings: {time.time() - _t0:.0f} s")
    net_d, _, _ = train(False, n_iter=0)
    print(f"parameters of the network: {sum(q.numel() for q in net_d.parameters())}")

    from scipy.optimize import curve_fit
    t1 = time.time()
    p, c = curve_fit(damped_oscillator, t_obs, x_obs, p0=[1.0, 0.2, 6.0, 0.0])
    dt = time.time() - t1
    e = np.sqrt(np.diag(c))
    err = damped_oscillator(t_test, *p) - x_true
    print(f"curve_fit ({dt:.3f} s): gamma = {p[1]:.3f} +/- {e[1]:.3f}, omega = {p[2]:.3f} +/- {e[2]:.3f}; "
          f"RMSE t<=3 {np.sqrt(np.mean(err[t_test <= 3]**2)):.3f}, t>3 {np.sqrt(np.mean(err[t_test > 3]**2)):.3f}")
    print(f"size of the ODE term relative to x: omega_0^4 = {6.007**4:.0f}")

    # the PINN (last net trained) over the prediction range, split in two halves
    with torch.no_grad():
        x_pinn = net(torch.tensor(t_test, dtype=torch.float32)[:, None]).numpy().ravel()
    for a, b in [(3, 4.5), (4.5, 6)]:
        k = (t_test > a) & (t_test <= b)
        print(f"PINN, {a} < t <= {b} s: RMSE {np.sqrt(np.mean((x_pinn[k] - x_true[k])**2)):.3f} m, "
              f"max |x| {np.abs(x_pinn[k]).max():.2f} m (truth {np.abs(x_true[k]).max():.2f} m)")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t_test, x_true, 'k', lw=1, label='truth')
    ax.plot(t_obs, x_obs, 'o', ms=4, label='30 data points')
    with torch.no_grad():
        ax.plot(t_test, net(torch.tensor(t_test, dtype=torch.float32)[:, None]).numpy().ravel(), label='PINN')
    ax.plot(t_test, damped_oscillator(t_test, *p), '--', label='curve_fit')
    ax.axvspan(3, 6, color='grey', alpha=0.15); ax.set_xlabel('t (s)'); ax.set_ylabel('x (m)'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig('outputs/ch10_pinn.png', dpi=100)
    print("saved outputs/ch10_pinn.png")
