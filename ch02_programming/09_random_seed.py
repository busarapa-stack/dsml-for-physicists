"""Ch.2  The modern NumPy random Generator versus the legacy global seed.

Book references: ssec:seed, code:seed-modern
"""
import numpy as np

# recommended: Generator object
rng = np.random.default_rng(seed=42)
samples = rng.normal(loc=0, scale=1, size=1000)
samples_new = samples

# legacy: hidden global state (avoid in new code)
np.random.seed(42)
samples = np.random.normal(loc=0, scale=1, size=1000)
samples_legacy = samples
samples = samples_new

print(f"Generator(seed=42): mean {samples.mean():+.5f}, first {samples[0]:+.5f}")
print(f"legacy seed(42)   : mean {samples_legacy.mean():+.5f}, first {samples_legacy[0]:+.5f}")
print("-> the same seed number gives DIFFERENT numbers in the two systems;"
      " a seed is only meaningful together with the generator that uses it")


# why the global state is fragile: a library call in between changes your numbers
def analysis_legacy():
    np.random.seed(1)
    a = np.random.normal(size=3)
    _ = np.random.random()                 # e.g. a helper function that also draws
    b = np.random.normal(size=3)
    return b


def analysis_generator():
    rng_data = np.random.default_rng(1)
    rng_other = np.random.default_rng(2)   # the helper gets its own generator
    a = rng_data.normal(size=3)
    _ = rng_other.random()
    b = rng_data.normal(size=3)
    return b


def analysis_generator_no_helper():
    rng_data = np.random.default_rng(1)
    a = rng_data.normal(size=3)
    b = rng_data.normal(size=3)
    return b


print("\nadding one extra random draw in a helper function:")
print(f"  separate Generators: b unchanged = "
      f"{np.array_equal(analysis_generator(), analysis_generator_no_helper())}")
np.random.seed(1); np.random.normal(size=3); b_ref = np.random.normal(size=3)
print(f"  global legacy state: b unchanged = {np.array_equal(analysis_legacy(), b_ref)}")
