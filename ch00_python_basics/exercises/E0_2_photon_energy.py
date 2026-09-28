"""E0.2  Energy of a green laser photon (532 nm) in joules and electronvolts."""
h = 6.62607015e-34        # Planck constant [J s], exact
c = 299792458.0           # speed of light [m/s], exact
e = 1.602176634e-19       # elementary charge [C], exact
lam_m = 532e-9            # wavelength [m]
E_J = h * c / lam_m
print(f"E = {E_J:.3g} J = {E_J / e:.3g} eV")
