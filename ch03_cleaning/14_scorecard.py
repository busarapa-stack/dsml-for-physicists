"""Ch.3  The data-quality scorecard of code:scorecard as reusable code (src/scorecard.py).

Book references: ssec:scorecard, code:scorecard, tab:quality-4
Prints the empty template (same items and maxima as the book: 9 + 9 + 6 + 3 + 9 = 36).
A filled-in example for the Galaxy10-style table is in exercises/E3_7_scorecard.py.
"""
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, ".")
from src.scorecard import ITEMS, Scorecard

print(Scorecard("<dataset name>").report())
print(f"\nsections: {', '.join(f'{k} /{3 * len(v)}' for k, v in ITEMS.items())}")
