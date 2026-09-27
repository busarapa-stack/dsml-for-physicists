"""E3.7  Fill in the scorecard for the cleaned Galaxy10-style table, with evidence that is
computed, not typed. Scores that need human judgement are marked as such.
"""
import os
import sys
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ".")
from src.scorecard import Scorecard

df = pd.read_csv('data/processed/galaxy10_cleaned.csv')
sc = Scorecard("data/processed/galaxy10_cleaned.csv (SYNTHETIC)", reviewer="example")
miss = df['flag_no_redshift'].mean()
fz = df.groupby(pd.cut(df['r_mag'], [0, 17, 18, 19, 30]), observed=True)['flag_no_redshift'].mean()
sc.score("completeness", "missing values: type and cause identified", 3,
         f"{miss:.0%} missing z; rises with r_mag: {fz.round(2).tolist()} -> MAR")
sc.score("completeness", "sample selection bias assessed", 2,
         "faint galaxies under-represented after dropping; not corrected")
sc.score("completeness", "coverage of the parameter space", 1,
         f"r_mag {df['r_mag'].min():.1f}-{df['r_mag'].max():.1f} incl. flagged values")
sc.score("accuracy", "calibration source and date", 0, "synthetic table: none")
sc.score("accuracy", "systematic uncertainty estimated", 1, "only imputation sensitivity (E3.4)")
sc.score("accuracy", "at least one sanity check against known physics", 2,
         f"negative z flagged: {df['flag_negative_z'].sum()} rows")
sc.score("consistency", "units stated and consistent for every column", 2,
         "z dimensionless, r_mag mag; not stored in the file")
sc.score("consistency", "dtypes and naming consistent", 3,
         f"all numeric: {all(pd.api.types.is_numeric_dtype(df[c]) for c in ['redshift', 'r_mag', 'g_mag'])}")
sc.score("timeliness", "dataset version and last-update date", 1, "generator seed only")
sc.score("reproducibility", "cleaning script and seed committed to git", 3, "E3_2_clean_messy.py, SEED = 2026")
sc.score("reproducibility", "requirements.txt and environment documented", 3, "code/requirements.txt")
sc.score("reproducibility", "notebook passes Restart & Run All", 0, "scripts, no notebook")
print(sc.report())
print("\nweak points: (1) redshift missing not at random -> any mean z is biased (Ch.4);"
      " (2) no calibration information -> systematic errors cannot be quoted.")

# ---- second data set: AFM features from E3.5 --------------------------------------
feat = pd.read_csv('data/processed/afm_features.csv')
af = Scorecard("data/processed/afm_features.csv (SYNTHETIC)", reviewer="example")
af.score("completeness", "missing values: type and cause identified", 3,
         f"no missing values in {len(feat)} curves")
af.score("completeness", "sample selection bias assessed", 1, "15 curves, selection not described")
af.score("completeness", "coverage of the parameter space", 2,
         f"slope {feat.slope_nN_per_nm.min():.2f}-{feat.slope_nN_per_nm.max():.2f} nN/nm")
af.score("accuracy", "calibration source and date", 0, "cantilever k not calibrated (synthetic)")
af.score("accuracy", "systematic uncertainty estimated", 2,
         f"smoothing bias of adhesion: {(feat.adhesion_nN_sg31 - feat.adhesion_nN).mean():+.2f} nN")
af.score("accuracy", "at least one sanity check against known physics", 3,
         "work ~ 0.5 x adhesion x snap length (triangle)")
af.score("consistency", "units stated and consistent for every column", 3, "units in column names")
af.score("consistency", "dtypes and naming consistent", 3, "all float")
af.score("timeliness", "dataset version and last-update date", 1, "generator seed only")
af.score("reproducibility", "cleaning script and seed committed to git", 3, "E3_5_afm_features.py, src/features.py")
af.score("reproducibility", "requirements.txt and environment documented", 3, "code/requirements.txt")
af.score("reproducibility", "notebook passes Restart & Run All", 3, "notebooks/04_quality_scorecard.ipynb")
print("\n" + af.report())
print("\nweak points: (1) the spring constant k is not calibrated -> every force carries one"
      " common systematic error; (2) peak adhesion depends on the smoothing window.")
