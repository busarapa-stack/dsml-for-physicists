"""Build and execute notebooks/04_quality_scorecard.ipynb (tab:ch3-deliverables).
Run from the chapter folder:  python notebooks/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.cells = [
    md("# Data-quality scorecard for the Chapter 3 outputs\n"
       "**Purpose:** score `galaxy10_cleaned.csv` and `afm_features.csv` with the scorecard "
       "of Section 3.5, using evidence computed from the files. Data are SYNTHETIC."),
    md("## 1. Imports"),
    code("import sys\nfrom pathlib import Path\nimport pandas as pd\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\nfrom src.scorecard import Scorecard"),
    md("## 2. Data\nBoth files are produced by `exercises/E3_2_clean_messy.py` and "
       "`exercises/E3_5_afm_features.py`."),
    code("gal = pd.read_csv(ROOT / 'data/processed/galaxy10_cleaned.csv')\n"
         "afm = pd.read_csv(ROOT / 'data/processed/afm_features.csv')\n"
         "print(gal.shape, afm.shape)"),
    md("## 3. Evidence"),
    code("print('missing redshift:', round(gal['flag_no_redshift'].mean(), 3))\n"
         "print('usable rows:', int(gal['usable'].sum()))\n"
         "print(afm.describe().loc[['mean', 'std']].round(3))"),
    md("## 4. Scorecard (Galaxy10-style table)"),
    code("sc = Scorecard('galaxy10_cleaned.csv (SYNTHETIC)', reviewer='example')\n"
         "sc.score('completeness', 'missing values: type and cause identified', 3, 'MAR on r_mag')\n"
         "sc.score('reproducibility', 'notebook passes Restart & Run All', 3, 'this notebook')\n"
         "print(sc.report())"),
    md("## 5. Conclusion\nOnly two items are scored here as an example; complete the rest "
       "with evidence, as in `exercises/E3_7_scorecard.py`."),
]
NotebookClient(nb, timeout=120, kernel_name="python3",
               resources={"metadata": {"path": str(HERE)}}).execute()
nbf.write(nb, HERE / "04_quality_scorecard.ipynb")
print("wrote notebooks/04_quality_scorecard.ipynb (executed top to bottom)")
