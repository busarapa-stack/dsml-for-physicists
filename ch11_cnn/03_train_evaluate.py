"""Ch.11  Train the MLP and the CNN on the toy galaxies; test on centred and shifted images.

Book references: ssec:cnn-train-loop, ssec:cnn-results, code:cnn-train and code:cnn-eval (line for line), tab:cnn-confusion
Writes the trained weights data/MLP.pt and data/CNN.pt, used by 05 (transfer) and 06 (Grad-CAM).
Numbers quoted in the text: CNN 30 epochs "about three minutes", MLP 15 epochs "under ten seconds";
MLP test 0.928, shifted 0.393; CNN test 0.959, shifted 0.965; confusion matrix of tab:cnn-confusion;
MLP confuses edge-on as cigar 40 times.
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import book_namespace

os.chdir(HERE)
globals().update(book_namespace())                       # code:cnn-arch + code:galaxy-gen

# --- code:cnn-train ----------------------------------------------------------
def augment(xb):
    """Random 90-degree rotation and mirror flip: exact symmetries of the pixel grid."""
    xb = torch.rot90(xb, np.random.randint(4), dims=(2, 3))
    if np.random.rand() < 0.5:
        xb = torch.flip(xb, dims=(3,))
    return xb

def fit(model, X, y, X_val, y_val, epochs=30, lr=2e-3, aug=False, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X[:, None]), torch.tensor(y)),
        batch_size=64, shuffle=True)
    Xv = torch.tensor(X_val[:, None])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    val_acc = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            if aug:
                xb = augment(xb)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_acc.append(np.mean(model(Xv).argmax(1).numpy() == y_val))
    return val_acc

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X[:, None])), dim=1).numpy()

models = {}
for name, Model, lr, epochs in [('MLP', MLP, 1e-3, 15), ('CNN', GalaxyCNN, 2e-3, 30)]:
    torch.manual_seed(0)
    models[name] = Model()
    fit(models[name], X_train, y_train, X_val, y_val, epochs=epochs, lr=lr)
# --- code:cnn-eval -----------------------------------------------------------
from sklearn.metrics import confusion_matrix

rng = np.random.default_rng(5)
X_shift = np.array([np.roll(img, tuple(rng.integers(-12, 13, 2)), axis=(0, 1))
                    for img in X_test])                  # galaxies moved off-centre
for name, model in models.items():
    pred = predict(model, X_test).argmax(1)
    acc_shift = np.mean(predict(model, X_shift).argmax(1) == y_test)
    print(f"{name}: test {np.mean(pred == y_test):.3f}, shifted {acc_shift:.3f}")
    print(confusion_matrix(y_test, pred))
# --- end of listings ---------------------------------------------------------

if __name__ == '__main__':
    for name in models:
        torch.save(models[name].state_dict(), f'data/{name}.pt')
    print("saved data/MLP.pt, data/CNN.pt")
    # timing of one training of each model (same calls as in code:cnn-train)
    for name, Model, lr, epochs in [('MLP', MLP, 1e-3, 15), ('CNN', GalaxyCNN, 2e-3, 30)]:
        torch.manual_seed(0)
        t0 = time.time()
        fit(Model(), X_train, y_train, X_val, y_val, epochs=epochs, lr=lr)
        print(f"training time {name} ({epochs} epochs): {time.time() - t0:.0f} s")
