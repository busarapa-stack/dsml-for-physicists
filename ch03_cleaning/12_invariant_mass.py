"""Ch.3  Conservation law as a feature: four-lepton invariant mass (Higgs search).

Book references: ssec:conservation-feat, ex:invariant-mass, code:invmass
Events are SYNTHETIC: 400 H -> ZZ* -> 4l decays (m_H = 125 GeV, 1.8 GeV resolution)
plus 1,600 smooth background events, all boosted along the beam. The individual lepton
energies change from event to event with the boost; the invariant mass does not.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent)
events = pd.read_csv('data/raw/higgs_4l_events.csv')

# --- code:invmass -----------------------------------------------------------
E  = events[['E1',  'E2',  'E3',  'E4' ]].sum(axis=1)
px = events[['px1', 'px2', 'px3', 'px4']].sum(axis=1)
py = events[['py1', 'py2', 'py3', 'py4']].sum(axis=1)
pz = events[['pz1', 'pz2', 'pz3', 'pz4']].sum(axis=1)

m2 = E**2 - px**2 - py**2 - pz**2
events['M_inv'] = np.sqrt(np.clip(m2, 0, None))   # guard tiny negatives

# --- checks -----------------------------------------------------------------
sig = events['is_signal']
print(f"{len(events)} events ({sig.sum()} signal)")
print(f"signal M_inv: median {events.loc[sig, 'M_inv'].median():.2f} GeV, "
      f"std {events.loc[sig, 'M_inv'].std():.2f} GeV  (generated: 125 GeV, 1.8 GeV)")
print(f"total energy E of signal events ranges {E[sig].min():.0f}-{E[sig].max():.0f} GeV "
      "-> E alone is not a good feature; M_inv is")
print(f"events with m2 < 0 before clipping: {(m2 < 0).sum()}")
win = events['M_inv'].between(120, 130)
print(f"window 120-130 GeV: {win.sum()} events, {100 * events.loc[win, 'is_signal'].mean():.0f} % signal")
pt1 = np.hypot(events['px1'], events['py1'])
print(f"transverse momentum of lepton 1: median {pt1.median():.1f} GeV (another conserved-quantity feature)")

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.arange(70, 300, 2.5)
ax.hist(events['M_inv'], bins=bins, histtype='step', color='k', label='all events')
ax.hist(events.loc[~sig, 'M_inv'], bins=bins, histtype='stepfilled', alpha=0.3,
        label='background only')
ax.set_xlabel(r'four-lepton invariant mass $M_{4\ell}$ [GeV]'); ax.set_ylabel('events / 2.5 GeV')
ax.set_title('SYNTHETIC H -> 4 leptons'); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('outputs/ch03_invariant_mass.png', dpi=150)
print("saved outputs/ch03_invariant_mass.png")
