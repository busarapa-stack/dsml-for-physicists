"""Ch.11  Deep ensemble of five CNNs; entropy of right and wrong predictions; stars and mergers (out of distribution).

Book references: ssec:cnn-ensemble, code:ens-ood (line for line), tab:model-card
Numbers quoted in the text: members 0.952-0.990 (a four-point spread from the seed alone); ensemble 0.979;
entropy 0.14 (correct) vs 0.68 (wrong); 83 % of test images with max probability > 0.9;
all 200 stars -> smooth cigar, 94 % above 0.9; 192/200 mergers -> smooth round, 88 % above 0.9.
"About fifteen minutes" for five members.
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import book_namespace, listing

os.chdir(HERE)
_ns = book_namespace()
listing(_ns, '03_train_evaluate.py', 'cnn-train', '\nmodels = {}')     # augment, fit, predict
globals().update(_ns)
_t0 = time.time()

# --- code:ens-ood ------------------------------------------------------------
def point_source(rng):                              # a star: PSF only, no extended light
    img = np.zeros((64, 64))
    c = (31.5 + rng.normal(0, 1.5, 2)).astype(int)
    img[c[0], c[1]] = rng.uniform(20, 60)
    img = np.arcsinh(gaussian_filter(img, 1.2) / 0.3)
    img = img / img.max() * rng.uniform(0.6, 1.0)
    return (img + rng.normal(0, 0.05, img.shape)).astype(np.float32)

def merger(rng):                                    # two smooth galaxies side by side
    a, b = make_galaxy(0, rng), make_galaxy(0, rng)
    return np.maximum(a, np.roll(b, rng.integers(10, 16), axis=1))

rng = np.random.default_rng(11)
X_star = np.array([point_source(rng) for _ in range(200)])
X_merge = np.array([merger(rng) for _ in range(200)])

members = []
for s in range(5):                                  # deep ensemble of 5 CNNs
    torch.manual_seed(s)
    members.append(GalaxyCNN())
    fit(members[-1], X_train, y_train, X_val, y_val, seed=s)
    print("member", s, np.mean(predict(members[-1], X_test).argmax(1) == y_test).round(3))

def ensemble(X):
    P = np.mean([predict(m, X) for m in members], axis=0)
    return P, -(P * np.log(P + 1e-12)).sum(axis=1)   # mean probability, entropy

P, H = ensemble(X_test)
ok = P.argmax(1) == y_test
print("ensemble accuracy", ok.mean().round(3),
      "| entropy: correct", H[ok].mean().round(3), "wrong", H[~ok].mean().round(3))
print("test images with max prob > 0.9:", np.mean(P.max(1) > 0.9).round(3))
for name, X in [('stars', X_star), ('mergers', X_merge)]:
    Pq, Hq = ensemble(X)
    print(name, "max prob > 0.9:", np.mean(Pq.max(1) > 0.9).round(2),
          "| predicted classes:", np.bincount(Pq.argmax(1), minlength=5))
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    print(f"five members trained and evaluated in {time.time() - _t0:.0f} s")
