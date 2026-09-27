"""Ch.7  Silhouette score for K = 2..7 on the blobs of code:kmeans.

Book references: ssec:silhouette, code:silhouette (line for line)
The listing continues the notebook of 01_kmeans_elbow.py (X_scaled, KMeans, np), which is executed first.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
_src = (HERE / '01_kmeans_elbow.py').read_text()
exec(_src[_src.index('\n# --- code:'):_src.index('\n# --- end of listings')])   # earlier listings

# --- code:silhouette ---------------------------------------------------------
from sklearn.metrics import silhouette_score, silhouette_samples

sil_scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_k = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels_k))
best_k = 2 + np.argmax(sil_scores)
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    for k, s in zip(range(2, 8), sil_scores):
        print(f"K = {k}: silhouette {s:.3f}")
    print("best K =", best_k)
