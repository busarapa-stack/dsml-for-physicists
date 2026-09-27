"""Restart & Run All from the command line: run a notebook top to bottom in a fresh
kernel and report the first error (the rule of thumb in ssec:hidden-state).

    python check_notebook.py messy_notebook.ipynb
    python check_notebook.py clean_notebook.ipynb
Needs: pip install nbformat nbclient ipykernel
"""
import re
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

path = Path(sys.argv[1] if len(sys.argv) > 1 else "messy_notebook.ipynb")
nb = nbf.read(path, as_version=4)
counts = [c.get("execution_count") for c in nb.cells if c.cell_type == "code"]
print(f"{path.name}: execution counts saved in the file = {counts}")
if counts != sorted(c for c in counts if c is not None) or None in counts:
    print("  -> saved outputs were NOT produced top to bottom (hidden-state warning)")
try:
    NotebookClient(nb, timeout=120, kernel_name="python3").execute()
    print("Restart & Run All: PASSED")
except CellExecutionError as e:
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(e))          # strip colour codes
    lines = [ln for ln in text.splitlines() if ln.strip()]
    err = next((ln for ln in reversed(lines) if "Error" in ln), lines[-1])
    idx = next(i for i, c in enumerate(nb.cells)
               if c.cell_type == "code" and c.get("outputs")
               and any(o.get("output_type") == "error" for o in c.outputs))
    print(f"Restart & Run All: FAILED at cell {idx}: {err.strip()}")
    sys.exit(1)
