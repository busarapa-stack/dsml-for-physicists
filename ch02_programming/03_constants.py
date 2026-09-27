"""Ch.2  Physical constants from scipy.constants, and what 'standard gravity' means.

Book references: ssec:scipy-const, code:scipy-const, exercise E2.1 answer
g = 9.80665 m/s^2 is a defined standard value, not the local gravity.
"""
import numpy as np
import scipy.constants as const

print(f"g   = {const.g} m/s^2")
print(f"h   = {const.h:.6e} J s")
print(f"c   = {const.c} m/s")
print(f"e   = {const.e:.6e} C")
print(f"k_B = {const.k:.6e} J/K")
print(f"N_A = {const.Avogadro:.6e} 1/mol")

# every constant carries its unit and uncertainty in physical_constants
for name in ("speed of light in vacuum", "electron mass", "Newtonian constant of gravitation"):
    val, unit, unc = const.physical_constants[name]
    print(f"{name:34s} {val:.10e} {unit:12s} +- {unc:.1e}")


def normal_gravity(lat_deg):
    """International gravity formula (GRS80 form), sea level [m/s^2]."""
    s = np.sin(np.deg2rad(lat_deg))
    s2 = np.sin(np.deg2rad(2 * lat_deg))
    return 9.780327 * (1 + 0.0053024 * s**2 - 0.0000058 * s2**2)


print("\nlocal gravity versus the standard value:")
for place, lat in [("equator", 0.0), ("Bangkok", 13.75), ("Kamphaeng Saen", 14.02),
                   ("pole", 90.0)]:
    gl = normal_gravity(lat)
    print(f"  {place:15s} lat {lat:5.2f} deg : g = {gl:.4f} m/s^2 "
          f"({100 * (gl - const.g) / const.g:+.2f} % vs standard)")
