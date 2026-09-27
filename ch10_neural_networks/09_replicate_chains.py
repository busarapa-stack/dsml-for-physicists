"""Ch.10  How much do the polymer results depend on the particular simulated ensemble?

The chain of code:chain-gen moves in continuous space, so its Metropolis trajectory is chaotic with respect to
floating-point rounding: a different CPU or numpy build (e.g. Apple Silicon vs Linux x86) flips one accept/reject
decision sooner or later and from then on produces a different - but statistically equivalent - ensemble.
(The Ising configurations of code:ising-configs are bit-identical everywhere: spins are integers.)
This script repeats code:chain-gen with four other seed sets (100 + k + 1000 r, r = 1..4; r = 0 is the book's run)
and reports the quantities the text quotes, to show the run-to-run spread a reader should expect.
About 10 minutes on two cores (two simulations at a time).
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import section, chain_data

os.chdir(HERE)
GEN = section(HERE / '03_polymer_chain.py', '\n# --- code:chain-gen', '\nT_grid = ')
T_grid = np.round(np.arange(0.6, 3.41, 0.2), 1)


def simulate(job):
    r, k, T = job
    g = {'np': np}
    exec(GEN, g)
    return g['make_chain_ensemble'](T=T, seed=100 + k + 1000 * r)


def analyse(conf, ener, T_lab):
    from scipy.optimize import curve_fit
    torch.set_num_threads(1)
    nb = {'np': np, 'torch': torch, 'nn': nn}
    exec(section(HERE / '04_polymer_classifier.py', '\n# --- code:polymer-nn', 'model, X_all, P, acc ='), nb)
    iu, ju = np.triu_indices(30, k=1)
    dist = np.linalg.norm(conf[:, iu] - conf[:, ju], axis=2)
    rg2 = np.mean(np.sum(conf**2, axis=2), axis=1)
    mean = np.array([rg2[T_lab == T].mean() for T in T_grid])
    f = lambda T, a, b, Ts, w: a + b * np.tanh((T - Ts) / w)
    p, c = curve_fit(f, T_grid, mean, p0=[5, 3, 1.8, 0.5], maxfev=20000)
    ts = [nb['crossing'](nb['train_phase_classifier'](dist, T_lab, 1.0, 3.0, seed=s)[2], T_lab, T_grid)[0] for s in range(5)]
    _, _, P, acc = nb['train_phase_classifier'](dist, T_lab, 1.0, 3.0)
    Pm = nb['crossing'](P, T_lab, T_grid)[1]
    cv06 = ener[T_lab == 0.6].var() / 0.36 / 30
    return p[2], np.sqrt(c[2, 2]), p[3], np.array(ts), Pm, cv06, mean


if __name__ == '__main__':
    rows = []
    book = chain_data()
    rows.append((0,) + analyse(book['conf'], book['ener'], book['T_lab']))
    jobs = [(r, k, T) for r in (1, 2, 3, 4) for k, T in enumerate(T_grid)]
    with Pool(2) as pool:
        res = pool.map(simulate, jobs)
    T_lab = np.repeat(T_grid, 400)
    for r in (1, 2, 3, 4):
        d = [res[i] for i, j in enumerate(jobs) if j[0] == r]
        conf = np.concatenate([x[0] for x in d]); ener = np.concatenate([x[1] for x in d])
        rows.append((r,) + analyse(conf, ener, T_lab))
    print(" r  tanh T*        w     T*_NN seeds 0-4 (mean)            P(1.2) P(1.4) P(1.6)  Cv/N(0.6)  <Rg2>(0.6, 3.4)")
    for r, Ts, eTs, w, ts, Pm, cv, mean in rows:
        print(f"{r:2d}  {Ts:.2f} +/- {eTs:.2f}  {w:.2f}  {np.round(ts, 2)} ({ts.mean():.2f})  "
              f"{Pm[3]:.2f}   {Pm[4]:.2f}   {Pm[5]:.2f}    {cv:.2f}      {mean[0]:.2f}, {mean[-1]:.2f}")
    allT = np.array([x[1] for x in rows]); allNN = np.array([x[4].mean() for x in rows])
    print(f"across the five ensembles: tanh T* {allT.min():.2f}-{allT.max():.2f} (SD {allT.std(ddof=1):.2f}); "
          f"T*_NN {allNN.min():.2f}-{allNN.max():.2f} (SD {allNN.std(ddof=1):.2f}); "
          f"gap tanh - NN {np.min(allT - allNN):.2f}-{np.max(allT - allNN):.2f}")
