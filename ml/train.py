# ml/train.py
"""Phase 3: train the LSTM, evaluate on the test split, compare to baseline."""
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from preprocess import build, make_windows, LOOKBACK, HORIZONS
from model import LatencyLSTM
from common import NETWORKS, Scaler

OUT = Path("shared-data")
torch.manual_seed(0)
np.random.seed(0)

# ---- data -----------------------------------------------------------------
wide = build()
X, Y, feat_cols, target_cols = make_windows(wide)
n = len(X);
a, b = int(n * 0.7), int(n * 0.85)
Xtr, Xva, Xte = X[:a], X[a:b], X[b:]
Ytr, Yva, Yte = Y[:a], Y[a:b], Y[b:]

# ---- scaling (fit on TRAIN only) ------------------------------------------
nf = X.shape[2]
xs = Scaler().fit(Xtr.reshape(-1, nf))
ys = Scaler().fit(Ytr)


def sx(A):
    return xs.transform(A.reshape(-1, nf)).reshape(A.shape).astype("float32")


def sy(A):
    return ys.transform(A).astype("float32")


Xtr_, Xva_, Xte_ = sx(Xtr), sx(Xva), sx(Xte)
Ytr_, Yva_, Yte_ = sy(Ytr), sy(Yva), sy(Yte)

# ---- train with early stopping --------------------------------------------
dl = DataLoader(TensorDataset(torch.tensor(Xtr_), torch.tensor(Ytr_)),
                batch_size=64, shuffle=True)
model = LatencyLSTM(nf)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.MSELoss()
Xva_t, Yva_t = torch.tensor(Xva_), torch.tensor(Yva_)

best, wait = 1e9, 0
for epoch in range(100):
    model.train()
    for xb, yb in dl:
        opt.zero_grad()
        lossf(model(xb), yb).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        vl = lossf(model(Xva_t), Yva_t).item()
    if vl < best - 1e-4:
        best, wait = vl, 0
        torch.save(model.state_dict(), OUT / "model.pt")
    else:
        wait += 1
        if wait >= 8:  # patience
            break
    print(f"epoch {epoch:02d}  val_mse={vl:.4f}")

# ---- evaluate on the test split in real (ms) units ------------------------
model.load_state_dict(torch.load(OUT / "model.pt"))
model.eval()
with torch.no_grad():
    pred = ys.inverse_transform(model(torch.tensor(Xte_)).numpy())
true = Yte  # already in ms

# naive persistence: predict "future baseline = latest observed baseline"
last = Xte[:, -1, :len(NETWORKS)]  # base_3G/4G/5G at time t
naive = np.concatenate([last] * len(HORIZONS), axis=1)


def mae(p, t):  return float(np.mean(np.abs(p - t)))


def rmse(p, t): return float(np.sqrt(np.mean((p - t) ** 2)))


print("\n=== Test metrics (ms) ===")
print(f"{'':10}{'LSTM MAE':>10}{'Naive MAE':>11}"
      f"{'LSTM RMSE':>11}{'Naive RMSE':>12}")
for hi, h in enumerate([10, 20, 30]):
    for ni, net in enumerate(NETWORKS):
        col = hi * len(NETWORKS) + ni
        print(f"+{h}m {net:4}"
              f"{mae(pred[:, col], true[:, col]):10.2f}"
              f"{mae(naive[:, col], true[:, col]):11.2f}"
              f"{rmse(pred[:, col], true[:, col]):11.2f}"
              f"{rmse(naive[:, col], true[:, col]):12.2f}")
print(f"\nOVERALL   LSTM MAE={mae(pred, true):.2f}   "
      f"Naive MAE={mae(naive, true):.2f}")

# ---- save scalers + metadata for the service ------------------------------
joblib.dump({"xs": xs, "ys": ys, "feat_cols": feat_cols,
             "lookback": LOOKBACK, "horizons": HORIZONS},
            OUT / "scalers.joblib")
print("\nSaved model.pt and scalers.joblib to shared-data/")
