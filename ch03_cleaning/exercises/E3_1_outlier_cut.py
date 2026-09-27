"""E3.1  What happens when 50 'abnormal' high-energy events are cut without checking.

A SYNTHETIC toy: 950 events follow the theory (exponential energy spectrum, mean 10 GeV)
and 50 come from a real extra process (a peak at 45 GeV) that the theory does not include.
Cutting the 50 makes the data 'agree with theory' -- and throws away the new physics.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(31)
theory_mean = 10.0
E = np.concatenate([rng.exponential(theory_mean, 950), rng.normal(45, 2, 50)])

cut = np.sort(E)[:-50]                              # remove the 50 highest energies
for name, x in (("all 1000 events", E), ("after cutting 50", cut)):
    ks = stats.kstest(x, "expon", args=(0, theory_mean))
    print(f"{name:17s}: mean {x.mean():6.2f} GeV, std {x.std(ddof=1):6.2f} GeV, "
          f"KS test vs theory p = {ks.pvalue:.3g}")
print(f"theory: mean = std = {theory_mean} GeV")
removed = np.sort(E)[-50:]
print(f"the 50 removed events: {removed.min():.1f}-{removed.max():.1f} GeV, "
      f"{np.mean(np.abs(removed - 45) < 6):.0%} of them within 6 GeV of 45 GeV "
      "-> a peak, not random noise")
print("the 'agreement' after the cut was produced by the cut itself (circular reasoning)")
