"""E6.5  Four colour-colour diagrams and the confusion matrices (with / without redshift) as figures.
Numbers are printed by 07_sdss_compare.py.  Uses the real table if present, else the SYNTHETIC stand-in."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
__file__ = str(HERE / '06_sdss_colors.py')          # the listing's os.chdir uses __file__
src = (HERE / '06_sdss_colors.py').read_text()
exec(src[:src.index('\n# --- end of listings')].split('"""', 2)[2])
print(Path('data/raw/SOURCE.txt').read_text().strip())

pairs = [('u_g', 'g_r'), ('g_r', 'r_i'), ('r_i', 'i_z'), ('u_g', 'r_i')]
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, (a, b) in zip(axes, pairs):
    for cls, color in [('STAR', 'tab:blue'), ('GALAXY', 'tab:red'), ('QSO', 'tab:green')]:
        sub = df[df['class'] == cls].sample(2000, random_state=1)
        ax.scatter(sub[a], sub[b], c=color, s=4, alpha=0.3, label=cls)
    ax.set_xlabel(a.replace('_', ' - ')); ax.set_ylabel(b.replace('_', ' - '))
    ax.set_xlim(*df[a].quantile([0.005, 0.995])); ax.set_ylim(*df[b].quantile([0.005, 0.995]))  # ignore extreme colours
axes[0].legend(); fig.tight_layout(); fig.savefig('outputs/E6_5_color_color_4.png', dpi=90)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, feats, title in [(axes[0], ['u_g', 'g_r', 'r_i', 'i_z', 'r', 'redshift'], 'with redshift'),
                         (axes[1], ['u_g', 'g_r', 'r_i', 'i_z', 'r'], 'without redshift')]:
    Xtr, Xte, ytr, yte = train_test_split(df[feats].values, df['class'].to_numpy(), test_size=0.25,
                                          stratify=df['class'].to_numpy(), random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    ConfusionMatrixDisplay.from_predictions(yte, rf.predict(Xte), labels=['GALAXY', 'STAR', 'QSO'],
                                            ax=ax, colorbar=False)
    ax.set_title(title)
fig.tight_layout(); fig.savefig('outputs/E6_5_confusion.png', dpi=110)
print("saved outputs/E6_5_color_color_4.png and outputs/E6_5_confusion.png")
