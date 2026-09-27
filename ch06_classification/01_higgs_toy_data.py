"""Ch.6  Toy H->bb events vs background, and a stratified train/validation split.

Book references: ssec:higgs-features, ssec:fs-PI, code:higgs-data, code:higgs-pipe (line for line)
All events are SYNTHETIC toy data (shapes illustrative only), seed 2026.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
Path('data').mkdir(exist_ok=True)

# --- code:higgs-data ---------------------------------------------------------
import numpy as np
import pandas as pd

def make_higgs_toy(n_sig=4000, n_bkg=16000, seed=2026):
    """Toy H->bb events vs background (GeV). Shapes are illustrative only."""
    rng = np.random.default_rng(seed)
    # signal: resonance at 125 GeV with ~10% resolution
    m_s   = rng.normal(125.0, 12.0, n_sig)
    pT_s  = rng.gamma(shape=4.0, scale=40.0, size=n_sig)     # mean 160 GeV
    dR_s  = 2 * m_s / np.maximum(pT_s, 60.0) * rng.normal(1.0, 0.15, n_sig)
    met_s = rng.exponential(20.0, n_sig)                       # semileptonic b
    # background: falling mass spectrum, softer pT, wider angles
    m_b   = 40.0 + rng.exponential(70.0, n_bkg)
    pT_b  = rng.gamma(shape=3.0, scale=40.0, size=n_bkg)     # mean 120 GeV
    dR_b  = rng.uniform(0.4, 4.0, n_bkg)
    met_b = rng.exponential(30.0, n_bkg)                       # e.g. ttbar
    df = pd.DataFrame({
        'm_bb':   np.concatenate([m_s, m_b]),
        'pT':     np.concatenate([pT_s, pT_b]),
        'dR':     np.clip(np.concatenate([dR_s, dR_b]), 0.4, 4.0),
        'ETmiss': np.concatenate([met_s, met_b]),
        'label':  np.concatenate([np.ones(n_sig, int), np.zeros(n_bkg, int)]),
    })
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

df = make_higgs_toy()
# --- code:higgs-pipe ---------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

features = ['m_bb', 'pT', 'dR', 'ETmiss']
X = df[features].values
y = df['label'].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    df.to_csv('data/higgs_toy.csv', index=False)
    print(f"{len(df)} events, signal fraction: all {y.mean():.3f}, train {y_train.mean():.3f}, "
          f"validation {y_val.mean():.3f}; validation size {len(y_val)} "
          f"({int(y_val.sum())} signal, {int((1 - y_val).sum())} background)")
    print("\nmean of each feature (GeV, dR dimensionless):")
    print(df.groupby('label')[features].mean().round(2).rename(index={0: 'background', 1: 'signal'}))
    s = df[df.label == 1]
    print(f"signal m_bb: 95 % of events in [{s.m_bb.quantile(0.025):.0f}, {s.m_bb.quantile(0.975):.0f}] GeV; "
          f"background fraction below/above that window: "
          f"{(df[df.label == 0].m_bb < 101).mean():.2f} / {(df[df.label == 0].m_bb > 149).mean():.2f}")
    r = s.dR * s.pT / (2 * s.m_bb)
    print(f"signal dR*pT/(2 m_bb): median {r.median():.2f} (background {(df[df.label == 0].eval('dR*pT/(2*m_bb)')).median():.2f})")
    print("saved data/higgs_toy.csv")
