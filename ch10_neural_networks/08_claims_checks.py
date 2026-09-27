"""Ch.10  Second-pass checks of statements in the text that have no listing of their own.

  1  all eight listings run in book order as one notebook, with the Chapter 4/5/6 code they continue from
     (instructor manual: "about six minutes"; polymer generation ~2 min, Ising ~1 min, PINN ~2 min)
  2  4x64 + 64 + 64x64 + 64 + 64x2 + 2 = 4,610 parameters, "several hundred times" logistic regression (5)
  3  backpropagation costs "about two to three forward passes"                               (Q2, ssec:backprop-algo)
  4  ReLU has zero second derivative almost everywhere, hence tanh in the PINN              (ssec:activation)
  5  E10.3 bug 1: without zero_grad the step keeps growing; bug 2: with softmax twice the best
     achievable probability is e/(1+e), so the loss cannot fall below -ln(e/(1+e))
  6  pair distances keep the full shape of the chain except its mirror image               (ssec:polymer-fe)
  7  fluctuation formula d<Rg^2>/dT = Cov(Rg^2, E)/T^2 agrees with finite differences of <Rg^2>(T)
  8  at the lowest temperature almost every pivot move is rejected                         (ssec:polymer-data)
  9  mc_dropout_predict leaves the network in evaluation mode                              (IM, corrected draft)
 10  crossing shifts: ~0.2 with the ends, ~0.2 with the features, ~0.6 from the <Rg^2> midpoint;
     7-20 times the ensemble spread 0.03                                                    (ssec:dl-uq-PI)
 11  a 128x128 image into 64 units needs more than a million weights                       (ssec:nn-bridge)
 12  PINN weight lambda: the ODE term is ~omega_0^4 ~ 10^3 times the data term; too small a lambda
     weakens the regularisation, too large biases gamma upward and drives the solution towards x = 0;
     no lambda removes the collapse of the amplitude after t = 4.5 s                         (notebox, ssec:pinn-example)
About 13 minutes (item 12 trains five PINNs of 30,000 iterations).
"""
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import section, higgs_split, ch4_metropolis, ch5_damped, CODE

os.chdir(HERE)
torch.set_num_threads(max(1, torch.get_num_threads()))

# 1 ------------------------------------------------------------------------------------------------
t0 = time.time()
nb = {'__name__': 'notebook'}
higgs_split(nb)
os.chdir(HERE)
buf = io.StringIO()
lap = {}
with contextlib.redirect_stdout(buf):
    for script, labels in [('01_higgs_mlp.py', ('mlp', 'train')), ('03_polymer_chain.py', ('chain-gen', 'chain-features')),
                           ('04_polymer_classifier.py', ('polymer-nn',)), ('05_ising_configs.py', ('ising-configs',)),
                           ('06_pinn_oscillator.py', ('pinn',)), ('07_mc_dropout_ensemble.py', ('mc-dropout',))]:
        if script == '05_ising_configs.py':
            ch4_metropolis(nb)
        if script == '06_pinn_oscillator.py':
            ch5_damped(nb)
        t1 = time.time()
        exec(section(HERE / script, f'\n# --- code:{labels[0]}', '\n# --- end of listings'), nb)
        lap[script] = time.time() - t1
out = buf.getvalue()
print(f"1  all eight listings ran as one notebook in {time.time() - t0:.0f} s; "
      + ', '.join(f"{k[:2]} {v:.0f} s" for k, v in lap.items()))
for key in ('AUC =', 'validation accuracy', 'T* of members', 'physics=True', 'gamma ='):
    print('   ', [l for l in out.splitlines() if l.strip().startswith(key)][0].strip())

# 2 ------------------------------------------------------------------------------------------------
n = 4 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2
print(f"2  parameters {n}; logistic regression on 4 features has 5 -> {n / 5:.0f} times")

# 3 ------------------------------------------------------------------------------------------------
model = nb['HiggsMLP']()
xb = nb['Xt'][:4096]; yb = nb['yt'][:4096]; loss_fn = nn.CrossEntropyLoss()
def tm(f, k=300):
    f(); t = time.perf_counter()
    for _ in range(k):
        f()
    return (time.perf_counter() - t) / k
with torch.no_grad():
    t_f = tm(lambda: loss_fn(model(xb), yb))
def fb():
    model.zero_grad(); loss_fn(model(xb), yb).backward()
t_fb = tm(fb)
print(f"3  forward {1e3 * t_f:.2f} ms, forward + backward {1e3 * t_fb:.2f} ms -> backward alone {(t_fb - t_f) / t_f:.1f}x a forward pass")

# 4 ------------------------------------------------------------------------------------------------
torch.manual_seed(0)
tt = torch.linspace(0, 6, 200)[:, None].requires_grad_(True)
for name, act in (('ReLU', nn.ReLU), ('tanh', nn.Tanh)):
    net = nn.Sequential(nn.Linear(1, 64), act(), nn.Linear(64, 64), act(), nn.Linear(64, 1))
    x = net(tt)
    dx = torch.autograd.grad(x.sum(), tt, create_graph=True)[0]
    ddx = torch.autograd.grad(dx.sum(), tt, allow_unused=True)[0]
    val = 0.0 if ddx is None else ddx.abs().max().item()
    print(f"4  {name}: max |d2x/dt2| over 200 points = {val:.3g}")

# 5 ------------------------------------------------------------------------------------------------
from torch.utils.data import DataLoader, TensorDataset
def grad_norms(zero_grad, n_steps=200):          # one epoch has 219 mini-batches
    torch.manual_seed(0)
    m = nb['HiggsMLP']()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(nb['Xt'], nb['yt']), batch_size=64, shuffle=True)
    norms, it = [], iter(loader)
    for step in range(n_steps):
        xb, yb = next(it)
        loss = loss_fn(m(xb), yb)
        if zero_grad:
            opt.zero_grad()
        loss.backward()
        norms.append(torch.sqrt(sum((p.grad**2).sum() for p in m.parameters())).item())
        opt.step()
    return np.array(norms)
g_ok, g_bug = grad_norms(True), grad_norms(False)
print(f"5  gradient norm at steps 10 / 100 / 200: correct {g_ok[9]:.2f} / {g_ok[99]:.2f} / {g_ok[-1]:.2f}; "
      f"no zero_grad {g_bug[9]:.1f} / {g_bug[99]:.1f} / {g_bug[-1]:.1f}")
pmax = np.e / (1 + np.e)
print(f"   softmax twice: best probability e/(1+e) = {pmax:.3f}, loss floor -ln = {-np.log(pmax):.3f} (observed final loss 0.366)")

# 6 ------------------------------------------------------------------------------------------------
from scipy.spatial.transform import Rotation
conf, T_lab, T_grid, ener = nb['conf'], nb['T_lab'], nb['T_grid'], nb['ener']
iu, ju = np.triu_indices(30, k=1)
r = conf[123]
D2 = np.zeros((30, 30)); D2[iu, ju] = nb['dist'][123]**2; D2 = D2 + D2.T
J = np.eye(30) - 1 / 30
w, V = np.linalg.eigh(-0.5 * J @ D2 @ J)                      # classical MDS
rec = V[:, -3:] * np.sqrt(w[-3:])
def procrustes(a, b, allow_reflection):
    U, _, Wt = np.linalg.svd(a.T @ b)
    R = U @ Wt
    if not allow_reflection and np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Wt
    return np.sqrt(np.mean(np.sum((a @ R - b)**2, axis=1)))
mirror = r * np.array([1, 1, -1])
d_mirror = np.linalg.norm(mirror[iu] - mirror[ju], axis=1)
print(f"6  shape rebuilt from the 435 distances (classical MDS): RMS error {procrustes(rec, r, True):.1e} with a reflection allowed, "
      f"{procrustes(rec, r, False):.2f} with rotations only; the mirror image has identical distances "
      f"(max diff {np.abs(d_mirror - nb['dist'][123]).max():.1e})")

# 7 ------------------------------------------------------------------------------------------------
rg2 = nb['rg2']
mean = np.array([rg2[T_lab == T].mean() for T in T_grid])
fluct = np.array([np.cov(rg2[T_lab == T], ener[T_lab == T])[0, 1] / T**2 for T in T_grid])
fd = np.gradient(mean, T_grid)
k = (T_grid >= 1.4) & (T_grid <= 2.2)
print(f"7  T = 1.4-2.2: fluctuation formula {fluct[k].min():.2f}-{fluct[k].max():.2f}, finite differences "
      f"{fd[k].min():.2f}-{fd[k].max():.2f}; correlation over all 15 T {np.corrcoef(fluct, fd)[0, 1]:.2f}")

# 8 ------------------------------------------------------------------------------------------------
src = section(HERE / '03_polymer_chain.py', '\n# --- code:chain-gen', '\nT_grid = ')
src = (src.replace('        if E_new <= E or rng.random() < np.exp(-beta * (E_new - E)):\n            r, E = new, E_new',
                   '        acc = E_new <= E or rng.random() < np.exp(-beta * (E_new - E))\n'
                   '        STATS[kind][0] += 1; STATS[kind][1] += acc\n'
                   '        if acc:\n            r, E = new, E_new')
          .replace('        if rng.random() < 0.5:                         # pivot move',
                   '        kind = "pivot" if rng.random() < 0.5 else "local"\n        if kind == "pivot":'))
assert 'STATS' in src and 'kind == "pivot"' in src
g = {'np': np}
exec(src, g)
for T in (0.6, 1.4, 3.4):
    g['STATS'] = {'pivot': [0, 0], 'local': [0, 0]}
    g['make_chain_ensemble'](T=T, n_samples=50, n_equil=500, seed=7)
    s = g['STATS']
    print(f"8  T = {T}: pivot acceptance {s['pivot'][1] / s['pivot'][0]:.3f}, crankshaft/end {s['local'][1] / s['local'][0]:.3f}")

# 9 ------------------------------------------------------------------------------------------------
m9 = nb['ensemble'][0][0]
nb['mc_dropout_predict'](m9, nb['X_all'][:10], n_samples=3)
print(f"9  after mc_dropout_predict: model.training = {m9.training}")

# 10 -----------------------------------------------------------------------------------------------
tab = {'dist': (1.25, 1.33, 1.43), 'rg2': (1.33, 1.55, 1.56), 'raw': (0.93, 1.32, 1.41)}
ends_shift = np.mean([np.ptp(v) for v in tab.values()])
feat_shift = np.mean([np.ptp([tab[f][i] for f in tab]) for i in range(3)])
print(f"10 mean shift across ends {ends_shift:.2f}, across features {feat_shift:.2f}; midpoint 1.90 - 1.33 = {1.90 - 1.33:.2f}; "
      f"0.2 / 0.03 = {0.2 / 0.03:.0f}, 0.6 / 0.03 = {0.6 / 0.03:.0f}")

# 11 -----------------------------------------------------------------------------------------------
print(f"11 128 x 128 x 64 = {128 * 128 * 64:,} weights")

# 12 -----------------------------------------------------------------------------------------------
train, t_test, x_true = nb['train'], nb['t_test'], nb['x_true']
t1 = time.time()
for lam in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
    net, gg, w0 = train(True, lam=lam)                        # 30,000 iterations as in the book
    with torch.no_grad():
        xp = net(torch.tensor(t_test, dtype=torch.float32)[:, None]).numpy().ravel()
    e = xp - x_true
    late = t_test > 4.5
    print(f"12 lambda {lam:.0e}: RMSE t<=3 {np.sqrt(np.mean(e[t_test <= 3]**2)):.3f}, "
          f"t>3 {np.sqrt(np.mean(e[t_test > 3]**2)):.3f}, max|x| for t>4.5 {np.abs(xp[late]).max():.2f} (truth 0.25), "
          f"gamma {gg:.3f}, omega_0 {w0:.3f}", flush=True)
print(f"   ({time.time() - t1:.0f} s)")
