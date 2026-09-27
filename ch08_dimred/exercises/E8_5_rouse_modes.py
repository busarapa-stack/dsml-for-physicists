"""E8.5  Rouse chain: eigenvalue triplets, mode shapes for p = 1, 2, tau_1, <Rg^2>, and the block-bootstrap CI.
Most numbers come from 04_rouse_pca.py and 05_eigenvalue_ci_block.py, which this script runs."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
for f in ('04_rouse_pca.py', '05_eigenvalue_ci_block.py'):
    print(f"--- {f}")
    subprocess.run([sys.executable, f], check=True)
