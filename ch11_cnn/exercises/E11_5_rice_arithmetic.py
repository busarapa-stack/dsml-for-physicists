"""E11.5  Arithmetic of the published rice-seed results (Kanchana et al. 2021): overall and balanced accuracy.

Per-variety recall 0.94 / 0.65 / 0.99 / 0.79 for KDML105 / RD23 / RD57 / PTT1; validation counts 65 / 77 / 90 / 80.
Answer in the text: 61, 50, 89, 63 correct = 263 / 312 = 84.3 %; balanced accuracy (0.938 + 0.649 + 0.989 + 0.788)/4 ~ 0.84.
"""
import numpy as np

n = np.array([65, 77, 90, 80]); rec = np.array([0.94, 0.65, 0.99, 0.79])
k = np.round(rec * n).astype(int)
print("correct per variety:", k, "total", k.sum(), "of", n.sum(), f"= {100 * k.sum() / n.sum():.1f} %")
print("recalls from the rounded counts:", np.round(k / n, 3), f"balanced accuracy {np.mean(k / n):.3f}")
print(f"training set 725 + validation 312 = {725 + 312} images")
