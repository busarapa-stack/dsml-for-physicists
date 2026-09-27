"""Ch.1  Read a nested JSON record and apply physics sanity checks to it.

Book references
    sssec:matproj, code:matproj-json, misconception box "database values are true values"

Checks
    formation energy of a pure element must be 0 (it is the reference state)
    energy_above_hull == 0  <=>  is_stable == True
    missing fields (null) are reported, never silently filled
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
rec = json.loads((HERE / "data" / "mp_like_mp-149.json").read_text(encoding="utf-8"))


def walk(d, prefix=""):
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            print(f"{prefix}{k}:")
            walk(v, prefix + "    ")
        else:
            print(f"{prefix}{k:28s} {v!r}")


walk(rec)

print("\nnested access: rec['symmetry']['number'] =", rec["symmetry"]["number"])

elements = {"Si"}
is_element = rec["formula_pretty"] in elements
print("\nsanity checks")
if is_element:
    ok = rec["formation_energy_per_atom"] == 0.0
    print(f"  pure element -> formation energy 0 : {'OK' if ok else 'FAIL'}")
ok = (rec["energy_above_hull"] == 0.0) == rec["is_stable"]
print(f"  e_above_hull == 0 <=> is_stable     : {'OK' if ok else 'FAIL'}")
missing = [k for k, v in rec.items() if v is None]
print(f"  fields still to fetch from the API  : {missing}")
print("\nreminder: band_gap from GGA-DFT is systematically lower than experiment;"
      " a model trained on it learns that bias too (veracity).")
