"""E9.2  Model selection with short data: the Chapter 5 oscillator (same parameters and seed) measured at 60 points in 3 s.
Uses the functions of 02_information_criteria.py.  About 1 minute (M4 multistart)."""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
__file__ = str(HERE / '02_information_criteria.py')
_s = (HERE / '02_information_criteria.py').read_text()
exec(_s[:_s.index("if __name__ == '__main__':")].split('"""', 2)[2])

rng = np.random.default_rng(seed=2026)
t_s = np.linspace(0, 3, 60)
x_s = damped_oscillator(t_s, 1.0, 0.3, 6.0, 0.5) + rng.normal(scale=0.05, size=t_s.size)
rows = table(t_s, x_s, 0.05, "60 points in the first 3 s, sigma 0.05 m")
