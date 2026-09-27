"""Ch.11  Transfer learning from the galaxy CNN to toy rice-grain images (25 per class) vs a size baseline.

Book references: ssec:transfer-pi, code:transfer (line for line)
Continues from code:cnn-train: models['CNN'] is loaded from data/CNN.pt (written by 03_train_evaluate.py).
Numbers quoted in the text: size features + logistic regression 0.80; scratch mean 0.31 (chance 0.25),
frozen 0.41, fine-tune 0.51 with seeds ranging 0.43-0.62.
About 1 minute.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import book_namespace

os.chdir(HERE)
Path('outputs').mkdir(exist_ok=True)
globals().update(book_namespace(trained=True))

# --- code:transfer -----------------------------------------------------------
SEED_TYPES = [(30, 8.0), (30, 9.0), (26, 8.0), (26, 9.5)]   # mean length, width (px)

def make_seed(cls, rng, size=64):
    """A toy rice-grain image: pointed ellipse with husk texture, random orientation."""
    L, W = rng.normal(SEED_TYPES[cls][0], 1.5), rng.normal(SEED_TYPES[cls][1], 0.5)
    y, x = np.mgrid[:size, :size] - (size - 1) / 2
    a = rng.uniform(0, np.pi)
    u, v = x * np.cos(a) + y * np.sin(a), -x * np.sin(a) + y * np.cos(a)
    inside = np.abs(2 * u / L)**2.2 + np.abs(2 * v / W)**2 < 1
    grain = inside * (0.7 + 0.3 * np.cos(np.pi * v / W)**2)
    grain *= 1 + 0.15 * gaussian_filter(rng.normal(size=grain.shape), 1.0)
    return (gaussian_filter(grain, 0.8) + rng.normal(0, 0.05, grain.shape)).astype(np.float32)

def make_seed_set(n_per_class, seed=0):
    rng = np.random.default_rng(seed)
    X = np.array([make_seed(c, rng) for c in range(4) for _ in range(n_per_class)])
    y = np.repeat(np.arange(4), n_per_class)
    p = rng.permutation(len(y))
    return X[p], y[p]

Xs_tr, ys_tr = make_seed_set(25, seed=21)          # only 100 labelled grains
Xs_te, ys_te = make_seed_set(200, seed=23)

def size_features(X):                              # length, width, area from image moments
    out = []
    for img in X:
        yy, xx = np.nonzero(img > 0.35)
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([xx, yy]))))[::-1]
        out.append([4 * np.sqrt(ev[0]), 4 * np.sqrt(ev[1]), len(xx)])
    return np.array(out)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
baseline.fit(size_features(Xs_tr), ys_tr)
print("size features + logistic regression:",
      np.mean(baseline.predict(size_features(Xs_te)) == ys_te).round(3))

for strategy in ('scratch', 'frozen', 'fine-tune'):
    accs = []
    for s in range(3):
        torch.manual_seed(s)
        model = GalaxyCNN(n_classes=4)
        if strategy != 'scratch':                  # start from the galaxy CNN
            model = GalaxyCNN()
            model.load_state_dict(models['CNN'].state_dict())
            model.head = nn.Linear(64, 4)          # new head for 4 seed types
            if strategy == 'frozen':
                for p in model.features.parameters():
                    p.requires_grad = False
        torch.manual_seed(s); np.random.seed(s)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-3)
        Xt, yt = torch.tensor(Xs_tr[:, None]), torch.tensor(ys_tr)
        for epoch in range(60):
            model.train()
            perm = np.random.permutation(len(yt))
            for b in range(0, len(yt), 32):
                i = perm[b:b + 32]
                loss = nn.functional.cross_entropy(model(Xt[i]), yt[i])
                opt.zero_grad(); loss.backward(); opt.step()
        accs.append(np.mean(predict(model, Xs_te).argmax(1) == ys_te))
    print(f"{strategy:9s}: test accuracy {np.round(accs, 3)}, mean {np.mean(accs):.3f}")
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(8, 2.3))
    for c in range(4):
        ax[c].imshow(Xs_tr[np.where(ys_tr == c)[0][0]], cmap='gray', origin='lower'); ax[c].axis('off')
        ax[c].set_title(f"type {c}: L {SEED_TYPES[c][0]}, W {SEED_TYPES[c][1]}", fontsize=8)
    fig.tight_layout(); fig.savefig('outputs/ch11_rice_examples.png', dpi=90)
    print("saved outputs/ch11_rice_examples.png")
