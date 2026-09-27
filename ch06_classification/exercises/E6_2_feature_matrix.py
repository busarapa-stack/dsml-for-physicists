"""E6.2  Class fractions and the pair plot of the four toy Higgs features, coloured by class."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import seaborn as sns

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
src = (HERE / '01_higgs_toy_data.py').read_text()
exec(src[src.index('\n# --- code:'):src.index('\n# --- end of listings')])

print(f"signal fraction: all {y.mean():.3f}, train {y_train.mean():.3f}, validation {y_val.mean():.3f}")
g = sns.pairplot(df.sample(3000, random_state=1), vars=features, hue='label',
                 plot_kws={'s': 6, 'alpha': 0.4}, diag_kind='hist')
g.savefig('outputs/E6_2_pairplot.png', dpi=90)
s = df[df.label == 1]
print(f"signal m_bb window (central 90 %): {s.m_bb.quantile(0.05):.0f}-{s.m_bb.quantile(0.95):.0f} GeV")
print(f"new feature dR*pT/(2 m_bb): signal median {(s.dR * s.pT / (2 * s.m_bb)).median():.2f}")
print("saved outputs/E6_2_pairplot.png")
