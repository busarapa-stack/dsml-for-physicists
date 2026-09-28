"""Ch.0  Lists, tuples and dictionaries holding physical measurements.

Book references: sec:py-collections, code:pendulum-list, code:particles-dict
"""
# --- code:pendulum-list ------------------------------------------------------
# time for 10 swings [s], pendulum length L = 0.80 m, ten trials
t10_s = [17.66, 18.02, 17.76, 17.93, 17.93, 18.28, 17.97, 17.97, 17.90, 18.29]

print(len(t10_s))          # number of trials
print(t10_s[0])            # first trial (index starts at 0)
print(t10_s[-1])           # last trial
print(t10_s[2:5])          # trials 3, 4 and 5 (the end index is excluded)
t10_s.append(18.02)        # an eleventh trial
print(len(t10_s))

position_m = (0.0, 1.5, -0.3)   # tuple: a point (x, y, z) that should not change

# --- code:particles-dict -----------------------------------------------------
particles = {
    "electron": {"mass_kg": 9.1093837e-31, "charge_C": -1.602176634e-19},
    "proton":   {"mass_kg": 1.67262192e-27, "charge_C": +1.602176634e-19},
    "muon":     {"mass_kg": 1.883531627e-28, "charge_C": -1.602176634e-19},
}
print(particles["muon"]["mass_kg"])
print(particles["muon"]["mass_kg"] / particles["electron"]["mass_kg"])

# --- end of listings ----------------------------------------------------------
print("[1, 2] * 2 =", [1, 2] * 2)
print("position_m =", position_m)
