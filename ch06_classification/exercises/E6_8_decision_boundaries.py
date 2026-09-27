"""E6.8  Decision boundaries of five classifiers on make_moons (500 points, noise 0.25)."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

os.chdir(Path(__file__).resolve().parent.parent)
X, y = make_moons(n_samples=500, noise=0.25, random_state=2026)
models = {'LogReg': LogisticRegression(), 'KNN k=1': KNeighborsClassifier(1),
          'KNN k=25': KNeighborsClassifier(25), 'SVM (RBF)': SVC(kernel='rbf'),
          'DecTree': DecisionTreeClassifier(max_depth=6, random_state=42)}
xx, yy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.5, 2, 300))
fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
for ax, (name, m) in zip(axes, models.items()):
    pipe = make_pipeline(StandardScaler(), m).fit(X, y)
    Z = pipe.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm'); ax.scatter(X[:, 0], X[:, 1], c=y, s=5, cmap='coolwarm')
    # count isolated "islands": connected regions of the minority prediction
    from scipy.ndimage import label
    n_regions = label(Z == 0)[1] + label(Z == 1)[1]
    ax.set_title(f"{name}\ntrain acc {pipe.score(X, y):.2f}, regions {n_regions}")
    print(f"{name:10s} training accuracy {pipe.score(X, y):.3f}, connected regions on the grid: {n_regions}")
fig.tight_layout(); fig.savefig('outputs/E6_8_decision_boundaries.png', dpi=110)
print("saved outputs/E6_8_decision_boundaries.png")
