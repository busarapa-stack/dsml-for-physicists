"""E0.3  Charge-to-mass ratio of three particles stored in a dictionary."""
particles = {
    "electron": {"mass_kg": 9.1093837e-31, "charge_C": -1.602176634e-19},
    "proton":   {"mass_kg": 1.67262192e-27, "charge_C": +1.602176634e-19},
    "muon":     {"mass_kg": 1.883531627e-28, "charge_C": -1.602176634e-19},
}
for name, p in particles.items():
    print(f"{name:9s} q/m = {p['charge_C'] / p['mass_kg']:+.4e} C/kg")
