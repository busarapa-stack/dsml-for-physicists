"""Ch.4  Exploratory data analysis in five steps on the (SYNTHETIC) Galaxy10-style features.

Book references: ssec:eda-workflow, code:eda
The listing is reproduced line for line; figures are saved instead of shown.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
os.chdir(Path(__file__).resolve().parent)

# --- code:eda ---------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/galaxy10_features.csv')

# 1: structure and summary numbers
df.info(); print(df.describe())

# 2: one variable at a time
sns.set_theme(style='whitegrid')
df.hist(bins=30, figsize=(10, 6))
plt.savefig('outputs/ch04_eda_1_hist.png', dpi=120)

# 3: outliers
plt.figure(); sns.boxplot(data=df[['r_mag', 'g_mag']])
plt.savefig('outputs/ch04_eda_2_box.png', dpi=120)

# 4: pairs of variables
sns.pairplot(df[['redshift', 'r_mag', 'g_mag', 'concentration']],
             diag_kind='hist', plot_kws={'alpha': 0.5, 's': 10})
plt.savefig('outputs/ch04_eda_3_pairs.png', dpi=100)

# 5: correlation heatmap
plt.figure()
corr = df.select_dtypes('number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.tight_layout(); plt.savefig('outputs/ch04_eda_4_corr.png', dpi=120)

print("\ncorrelation matrix:\n", corr.round(3))
print("saved outputs/ch04_eda_1_hist.png ... ch04_eda_4_corr.png")
