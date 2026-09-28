"""Ch.0  Variables, numeric types, scientific notation, f-strings and units.

Book references: sec:py-variables, code:gravity-moon, code:float-precision, code:ev-joule
"""
import math

# --- code:gravity-moon -------------------------------------------------------
G = 6.674e-11          # gravitational constant [N m^2 / kg^2]
M_earth = 5.972e24     # mass of the Earth [kg]
M_moon = 7.342e22      # mass of the Moon [kg]
r_m = 3.844e8          # mean Earth-Moon distance [m]

F_N = G * M_earth * M_moon / r_m**2
print(F_N)                          # full precision
print(f"F = {F_N:.3e} N")           # 3 decimals in scientific notation
print(type(F_N), type(10), type("10"))

# --- code:float-precision ----------------------------------------------------
a = 0.1 + 0.2
print(a)                            # 0.30000000000000004
print(a == 0.3)                     # False
print(math.isclose(a, 0.3))         # True: compare within a tolerance

# --- code:ev-joule ------------------------------------------------------------
e_C = 1.602176634e-19               # elementary charge [C], exact by definition
E_eV = 13.6                         # hydrogen ionization energy [eV]
E_J = E_eV * e_C                    # 1 eV = e x 1 V
print(f"{E_eV} eV = {E_J:.4e} J")
print(f"back to eV: {E_J / e_C:.1f} eV")

# --- end of listings ----------------------------------------------------------
print("7 / 2 =", 7 / 2, "  7 // 2 =", 7 // 2, "  7 % 2 =", 7 % 2, "  2**10 =", 2**10)
print("E0.1: 2 + 3 * 4 ** 2 / 8 =", 2 + 3 * 4 ** 2 / 8)
