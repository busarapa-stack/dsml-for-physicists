"""Ch.0  Reading error messages (tracebacks) of the most common mistakes.

Book references: sec:py-errors, code:traceback, exercise E0.8
Each mistake is run inside try/except so that this script keeps going;
in a notebook you would see the traceback directly.
"""
# --- code:traceback ----------------------------------------------------------
import traceback

def kinetic_energy(m_kg, v_ms):
    return 0.5 * m_kg * v_ms**2

speeds = ["3.0", "4.5"]            # read from a text file: these are strings
try:
    KE = kinetic_energy(2.0, speeds[0])
except TypeError:
    traceback.print_exc(limit=2)

# --- end of listings ----------------------------------------------------------
print("fixed:", kinetic_energy(2.0, float(speeds[0])), "J")
cases = {
    "NameError": "print(velocity)",
    "TypeError": "'g = ' + 9.78",
    "IndexError": "[1, 2, 3][3]",
    "ZeroDivisionError": "1 / 0",
    "KeyError": "{'mass_kg': 1.0}['mass']",
}
for name, code in cases.items():
    try:
        eval(code)
    except Exception as e:
        print(f"{name:18s} <- {code:28s} : {type(e).__name__}: {e}")
