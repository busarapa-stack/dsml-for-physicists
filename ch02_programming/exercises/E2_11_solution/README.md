# ch02-oscillator

Simulation of an undamped and a damped harmonic oscillator
(ω = 2π rad/s, γ = 0.1 s⁻¹) with two numerical checks:
relative energy drift of the undamped case, and γ recovered from the energy decay.

## Run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python sim.py
```

## Expected output

```
undamped: relative energy drift = 3.06e-08
damped:   gamma set = 0.10, gamma fit = 0.1001
saved ho_comparison.pdf
```

The figure `ho_comparison.pdf` has three panels: x(t), v(t), E(t)/m.
