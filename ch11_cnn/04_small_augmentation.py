"""Ch.11  100 images per class, 60 epochs, with and without rotation/flip augmentation.

Book references: ssec:augmentation, code:small-aug (line for line), notebox on selecting the best epoch
Numbers quoted in the text: without augmentation test 0.737, with 0.771 (full data 0.959);
best validation over 60 epochs 0.814 / 0.829, last epoch 0.749 / 0.778 ("5-8 points" above test).
About 1 minute.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared import book_namespace, listing

os.chdir(HERE)
_ns = book_namespace()
listing(_ns, '03_train_evaluate.py', 'cnn-train', '\nmodels = {}')     # augment, fit, predict
globals().update(_ns)

# --- code:small-aug ----------------------------------------------------------
small = np.concatenate([np.where(y_train == c)[0][:100] for c in range(5)])   # 100 per class
for aug in (False, True):
    torch.manual_seed(0)
    model = GalaxyCNN()
    val_acc = fit(model, X_train[small], y_train[small], X_val, y_val, epochs=60, aug=aug)
    test_acc = np.mean(predict(model, X_test).argmax(1) == y_test)
    print(f"augmentation={aug}: best val {max(val_acc):.3f}, "
          f"last val {val_acc[-1]:.3f}, test {test_acc:.3f}")
# --- end of listings ---------------------------------------------------------
