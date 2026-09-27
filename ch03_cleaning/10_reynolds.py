"""Ch.3  Buckingham pi in practice: the Reynolds number as one feature replacing three.

Book references: ssec:dim-analysis, ex:reynolds, code:reynolds
Check of the text: sorted by Re the two flow classes separate with one threshold,
while v alone or L alone does NOT separate them.
"""
import pandas as pd

# --- code:reynolds ----------------------------------------------------------
flow_df = pd.DataFrame({
    'v_ms':       [0.005, 0.10, 0.02, 0.015, 0.08],
    'L_m':        [0.10,  0.01, 0.50, 0.02,  0.08],
    'nu_m2s':     [1e-6] * 5,
    'flow_class': ['laminar', 'laminar', 'turbulent', 'laminar', 'turbulent'],
})
flow_df['Re'] = flow_df['v_ms'] * flow_df['L_m'] / flow_df['nu_m2s']
# Re = 500, 1000, 10000, 300, 6400

print(flow_df.to_string(index=False))


def separable(col):
    lam = flow_df.loc[flow_df.flow_class == 'laminar', col]
    tur = flow_df.loc[flow_df.flow_class == 'turbulent', col]
    return lam.max() < tur.min() or tur.max() < lam.min()


for col in ('v_ms', 'L_m', 'Re'):
    print(f"one threshold on {col:5s} separates the classes: {separable(col)}")
