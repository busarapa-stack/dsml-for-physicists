"""E3.3  Summarise r_mag and redshift by galaxy class; plot the mean with the STANDARD
ERROR of the mean (std / sqrt(n)), not the standard deviation.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)
df = pd.read_csv('data/processed/galaxy10_cleaned.csv')
use = df[df['usable']]
s = use.groupby('galaxy_class').agg(n_rows=('redshift', 'size'), n_z=('redshift', 'count'),
                                     mean_z=('redshift', 'mean'), std_z=('redshift', 'std'),
                                     mean_r=('r_mag', 'mean'))
s['sem_z'] = s['std_z'] / np.sqrt(s['n_z'])
miss = df.groupby('galaxy_class')['flag_no_redshift'].mean().rename('frac_missing_z')
s = s.join(miss).sort_values('mean_z')
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 20)
print(s.round(4))

fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(range(len(s)), s['mean_z'], yerr=s['sem_z'], fmt='o', capsize=3)
ax.set_xticks(range(len(s))); ax.set_xticklabels(s.index, rotation=40, ha='right', fontsize=8)
ax.set_ylabel('mean redshift ± SEM'); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('outputs/E3_3_mean_redshift_by_class.png', dpi=150)
print("\nsaved outputs/E3_3_mean_redshift_by_class.png")
print("with ~50 galaxies per class the SEM is ~0.006; differences between classes of that"
      " size are not evidence of a physical difference. In this synthetic table the classes"
      " were drawn with the same redshift distribution, so none is expected.")
