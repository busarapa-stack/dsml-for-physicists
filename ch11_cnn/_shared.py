"""Shared helpers for Chapter 11: run the book's listings in the order the chapter builds them up.

  code:cnn-arch    GalaxyCNN, MLP                       (01_cnn_architecture.py)
  code:galaxy-gen  make_galaxy, X_train ... y_test     (02_galaxy_images.py; cached in data/galaxies.npz)
  code:cnn-train   augment, fit, predict, models       (03_train_evaluate.py; trained weights in data/*.pt)
Later listings (small-aug, transfer, gradcam, ens-ood) continue from these names, so every script starts
with book_namespace(), which execs the same listing code, not a copy.
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


def listing(ns, script, label, stop='\n# --- end of listings'):
    exec(section(HERE / script, f'\n# --- code:{label}', stop), ns)
    return ns


def book_namespace(trained=False, quiet=True):
    """Namespace with code:cnn-arch and code:galaxy-gen (images from data/galaxies.npz when present).
    trained=True adds augment/fit/predict (code:cnn-train) and models['MLP'], models['CNN'] with the weights
    written by 03_train_evaluate.py (trained here if the files are missing, about 3 minutes)."""
    import torch
    ns = {'__name__': 'ch11', 'np': np}
    out = io.StringIO()
    with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
        listing(ns, '01_cnn_architecture.py', 'cnn-arch')
        f = HERE / 'data' / 'galaxies.npz'
        if f.exists():                                   # definitions only, then the cached images
            listing(ns, '02_galaxy_images.py', 'galaxy-gen', '\nX_train, y_train = ')
            d = np.load(f)
            ns.update({k: d[k] for k in ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test')})
        else:
            listing(ns, '02_galaxy_images.py', 'galaxy-gen')
            f.parent.mkdir(exist_ok=True)
            np.savez_compressed(f, **{k: ns[k] for k in ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test')})
    if trained:
        with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
            listing(ns, '03_train_evaluate.py', 'cnn-train', '\nmodels = {}')
        ns['models'] = {}
        for name in ('MLP', 'CNN'):
            w = HERE / 'data' / f'{name}.pt'
            if not w.exists():
                import subprocess, sys
                subprocess.run([sys.executable, str(HERE / '03_train_evaluate.py')], check=True)
            m = ns['GalaxyCNN']() if name == 'CNN' else ns['MLP']()
            m.load_state_dict(torch.load(w))
            m.eval()
            ns['models'][name] = m
    ns.pop('__name__', None)                             # keep the caller's __name__ after globals().update
    return ns
