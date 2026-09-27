# ml/service.py
"""Phase 5: serve LSTM forecasts and water-filling assignments over HTTP."""
import torch
import joblib
from pathlib import Path
from fastapi import FastAPI

from model import LatencyLSTM
from preprocess import build
from decision_engine import assign_devices
from common import NETWORKS, N_DEVICES

OUT = Path("shared-data")
bundle = joblib.load(OUT / "scalers.joblib")
xs, ys = bundle["xs"], bundle["ys"]
feat_cols, LOOKBACK = bundle["feat_cols"], bundle["lookback"]

model = LatencyLSTM(len(feat_cols))
model.load_state_dict(torch.load(OUT / "model.pt"))
model.eval()

wide = build()                       # full feature frame, for live windows
app = FastAPI(title="Network Switching ML Service")
# a moving cursor so repeated calls walk forward through time (demo effect)
state = {"cursor": LOOKBACK + len(wide) // 2, "prev": {}}


def forecast_at(i):
    win = wide[feat_cols].values[i - LOOKBACK:i].astype("float32")
    xw = xs.transform(win).reshape(1, LOOKBACK, len(feat_cols)).astype("float32")
    with torch.no_grad():
        p = ys.inverse_transform(model(torch.tensor(xw)).numpy())[0]  # 9 values
    plus30 = {NETWORKS[j]: round(float(p[6 + j]), 2) for j in range(3)}
    allh = {str(h): {NETWORKS[j]: round(float(p[k * 3 + j]), 2)
                     for j in range(3)}
            for k, h in enumerate([10, 20, 30])}
    return plus30, allh, str(wide.index[i])


@app.get("/predict")
def predict():
    p30, allh, ts = forecast_at(state["cursor"])
    return {"timestamp": ts, "forecast_plus30": p30, "forecast_all": allh}


@app.post("/assign")
def assign():
    i = state["cursor"]
    p30, allh, ts = forecast_at(i)
    devices = [{"id": f"dev{d:03d}", "priority": d < 5} for d in range(N_DEVICES)]
    final, log, counts = assign_devices(p30, devices, state["prev"])
    state["prev"] = final
    state["cursor"] = i + 3                       # advance one 30-min cycle
    if state["cursor"] >= len(wide):
        state["cursor"] = LOOKBACK
    return {"timestamp": ts, "forecast_plus30": p30, "counts": counts,
            "switches": sum(l["switched"] for l in log), "log": log[:20]}