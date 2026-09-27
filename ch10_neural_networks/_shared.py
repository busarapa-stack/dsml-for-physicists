"""Shared helpers for Chapter 10: pull listing code from earlier chapters and this one.

Every Chapter 10 listing continues from code the reader has already run:
  code:mlp           needs X_train, X_val, y_train, y_val   (Chapter 6, code:higgs-data + code:higgs-pipe)
  code:ising-configs needs init_lattice, metropolis_step    (Chapter 4, code:metropolis)
  code:pinn          needs damped_oscillator                (Chapter 5, code:damped-cf)
  code:polymer-nn    needs conf, ener, T_lab, T_grid, dist  (this chapter, code:chain-gen + code:chain-features)
The helpers below exec exactly those listing sections, so the scripts use the book code, not copies.
"""
import contextlib
import io
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent


def section(path, start, stop):
    s = Path(path).read_text(encoding='utf-8')
    s = s[s.index(start):]
    return s[:s.index(stop)]


def higgs_split(ns):
    """Chapter 6 toy Higgs data and the 70/30 stratified split."""
    ns.setdefault('__name__', 'ch06')
    ns['__file__'] = str(CODE / 'ch06_classification' / '01_higgs_toy_data.py')
    exec(section(ns['__file__'], '\n# --- code:higgs-data', '\n# --- end of listings'), ns)
    return ns


def ch4_metropolis(ns):
    """Chapter 4 init_lattice / metropolis_step (module-level rng seeded 2026)."""
    ns.setdefault('np', np)
    exec(section(CODE / 'ch04_statistics' / '07_metropolis_ising.py', '\n# --- code:metropolis', 'def simulate_T'), ns)
    return ns


def ch5_damped(ns):
    """Chapter 5 damped_oscillator (and the Chapter 5 data, which Chapter 10 does not use)."""
    ns.setdefault('np', np)
    with contextlib.redirect_stdout(io.StringIO()):          # the Chapter 5 fit printout is not needed here
        exec(section(CODE / 'ch05_regression' / '04_damped_oscillator.py', '\n# --- code:damped-cf', '\n# --- numbers quoted'), ns)
    return ns


def listing(ns, script, label, stop='\n# --- end of listings'):
    """Run one listing section of a Chapter 10 script in namespace ns."""
    exec(section(HERE / script, f'\n# --- code:{label}', stop), ns)
    return ns


def chain_data(ns=None):
    """Polymer ensemble of code:chain-gen, from data/chain.npz (written by 03_polymer_chain.py).
    If the file is missing the generator listing is run (about two minutes).
    Adds dist and rg2 exactly as in code:chain-features (without re-printing the table)."""
    ns = {} if ns is None else ns
    f = HERE / 'data' / 'chain.npz'
    if f.exists():
        d = np.load(f)
        for k in ('conf', 'ener', 'T_lab', 'T_grid'):
            ns[k] = d[k]
    else:
        listing(ns, '03_polymer_chain.py', 'chain-gen', '\n# --- code:chain-features')
        f.parent.mkdir(exist_ok=True)
        np.savez(f, conf=ns['conf'], ener=ns['ener'], T_lab=ns['T_lab'], T_grid=ns['T_grid'])
    conf = ns['conf']
    N = conf.shape[1]
    iu, ju = np.triu_indices(N, k=1)
    ns['N'] = N
    ns['dist'] = np.linalg.norm(conf[:, iu] - conf[:, ju], axis=2)
    ns['rg2'] = np.mean(np.sum(conf**2, axis=2), axis=1)
    return ns
