"""Download a DR17 table in the format of the Kaggle 'Stellar Classification Dataset - SDSS17'
from the SDSS SkyServer SQL service, with the same class counts, and save it as
data/raw/star_classification.csv (the synthetic stand-in there is replaced; a real file is kept).

Run on a computer with internet access:
    python optional/fetch_sdss_skyserver.py
Needs: requests, pandas.  Takes a few minutes; SkyServer limits each query to about 10 minutes.
The selection is not identical to the Kaggle file (whose selection is not documented), so numbers
will differ slightly; the class counts are matched.
"""
import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests

os.chdir(Path(__file__).resolve().parent.parent)
URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
COUNTS = {'GALAXY': 59445, 'STAR': 21594, 'QSO': 18961}
RAW = Path('data/raw/star_classification.csv')

SQL = """
SELECT TOP {n}
  p.objid AS obj_ID, p.ra AS alpha, p.dec AS delta,
  p.u, p.g, p.r, p.i, p.z,
  p.run AS run_ID, p.rerun AS rerun_ID, p.camcol AS cam_col, p.field AS field_ID,
  s.specobjid AS spec_obj_ID, s.class, s.z AS redshift,
  s.plate, s.mjd AS MJD, s.fiberid AS fiber_ID
FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE s.class = '{cls}' AND s.zWarning = 0
"""

if RAW.exists() and 'spec_obj_ID' in pd.read_csv(RAW, nrows=1).columns and '--force' not in sys.argv:
    sys.exit(f"{RAW} already holds a real table; use --force to download again")
parts = []
for cls, n in COUNTS.items():
    print(f"querying {n} {cls} ...", flush=True)
    r = requests.get(URL, params={'cmd': SQL.format(n=n, cls=cls), 'format': 'csv'}, timeout=900)
    r.raise_for_status()
    text = r.text
    if text.startswith('#Table'):                     # SkyServer prepends a table-name line
        text = text.split('\n', 1)[1]
    part = pd.read_csv(io.StringIO(text))
    print(f"  got {len(part)} rows")
    parts.append(part)
df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=2026).reset_index(drop=True)
RAW.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(RAW, index=False)
Path('data/raw/SOURCE.txt').write_text("REAL SDSS DR17 table from SkyServer (optional/fetch_sdss_skyserver.py)\n")
print(f"saved {RAW}: {len(df)} rows; class fractions:")
print(df['class'].value_counts(normalize=True).round(3))
