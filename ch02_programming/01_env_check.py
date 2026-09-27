"""Check that the scientific Python stack is installed and runs."""
# Ch.2  physics_env_check.py
#
# Book references: ssec:hello-physics, code:env-check, exercise E2.1
# Expected: every line prints without error and g = 9.80665 m/s^2.
# Version numbers differ from machine to machine; that is normal.
import sys
import numpy
import scipy
import matplotlib
import scipy.constants as const

print(f"Python version : {sys.version.split()[0]}")
print(f"NumPy version  : {numpy.__version__}")
print(f"SciPy version  : {scipy.__version__}")
print(f"Matplotlib ver.: {matplotlib.__version__}")
print()
print(f"g = {const.g} m/s^2 (standard gravity)")
print(f"h = {const.h:.6e} J s")
print(f"c = {const.c} m/s")
