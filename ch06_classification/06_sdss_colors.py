"""Ch.6  Colour-colour diagrams of the SDSS table (real file or synthetic stand-in, see 05_sdss_data.py).

Book references: ssec:sdss-eda, code:sdss-cc (reproduced line for line; the figure is saved)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

os.chdir(Path(__file__).resolve().parent)
if not Path('data/raw/star_classification.csv').exists():
    os.system('python 05_sdss_data.py')

# --- code:sdss-cc ------------------------------------------------------------
import matplotlib.pyplot as plt

df = pd.read_csv('data/raw/star_classification.csv')
bands = ['u', 'g', 'r', 'i', 'z']
df = df[(df[bands] > -100).all(axis=1)]          # drop sentinel magnitudes

df['u_g'] = df['u'] - df['g']                    # color indices remove distance
df['g_r'] = df['g'] - df['r']
df['r_i'] = df['r'] - df['i']
df['i_z'] = df['i'] - df['z']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for cls, color in [('STAR', 'tab:blue'), ('GALAXY', 'tab:red'), ('QSO', 'tab:green')]:
    sub = df[df['class'] == cls].sample(2000, random_state=1)
    axes[0].scatter(sub['u_g'], sub['g_r'], c=color, alpha=0.4, s=8, label=cls)
    axes[1].scatter(sub['g_r'], sub['r_i'], c=color, alpha=0.4, s=8, label=cls)
axes[0].set_xlabel('u - g'); axes[0].set_ylabel('g - r')
axes[1].set_xlabel('g - r'); axes[1].set_ylabel('r - i')
axes[0].set_xlim(-0.5, 4); axes[0].set_ylim(-0.5, 2.5)   # a few faint objects
axes[1].set_xlim(-0.5, 2.5); axes[1].set_ylim(-0.5, 1.5) # have extreme colours
axes[0].legend()
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    print(Path('data/raw/SOURCE.txt').read_text().strip())
    print(f"{len(df)} rows after dropping sentinel magnitudes")
    fig.savefig('outputs/ch06_sdss_color_color.png', dpi=110)
    print(df.groupby('class')[['u_g', 'g_r', 'r_i', 'i_z', 'redshift']].median().round(2))
    print("saved outputs/ch06_sdss_color_color.png")
