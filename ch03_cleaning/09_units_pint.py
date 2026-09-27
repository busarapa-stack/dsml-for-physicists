"""Ch.3  Carry units through a calculation with pint; unit mistakes fail loudly.

Book references: ssec:pint, code:pint (Mars Climate Orbiter: 1 lbf s = 4.45 N s)
Needs: pip install pint
"""
# --- code:pint --------------------------------------------------------------
import pint
import scipy.constants as const

ureg = pint.UnitRegistry()

v = 5.0 * ureg.meter / ureg.second
m = 2.0 * ureg.kilogram
KE = 0.5 * m * v**2
print(KE.to(ureg.joule))           # 25.0 joule

# attach units to a scipy constant
g = const.g * ureg.meter / ureg.second**2
h = 10.0 * ureg.meter
t_fall = (2 * h / g)**0.5
print(t_fall.to(ureg.second))      # about 1.428 second

# a unit mistake is caught immediately
try:
    bad = m + h                     # kg + m
except pint.errors.DimensionalityError as e:
    print(f"caught: {e}")

# --- the Mars Climate Orbiter factor ----------------------------------------
impulse = 1.0 * ureg.pound_force * ureg.second
print(f"\n1 lbf s = {impulse.to(ureg.newton * ureg.second):.4f}  -> the factor of about 4.45 in the text")

# --- AFM units used in this chapter -----------------------------------------
k = 0.1 * ureg.nanonewton / ureg.nanometer
print(f"0.1 nN/nm = {k.to(ureg.newton / ureg.meter)}")
work = (1 * ureg.nanonewton * ureg.nanometer).to(ureg.joule)
print(f"1 nN x 1 nm = {work:.0e} = {work.to(ureg.attojoule):.3f}")
