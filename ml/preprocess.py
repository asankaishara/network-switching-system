#	ml/preprocess.py
"""Phase	2:	build	model-ready	windows	and	a	chronological	split."""
import numpy as np
import pandas as pd
from pathlib import Path
from common import NETWORKS

OUT = Path("shared-data")
LOOKBACK = 12  # 12	steps	x	10	min	=	2	h	of	history	per	sample
HORIZONS = [1, 2, 3]  # predict	+10,	+20,	+30	min	ahead


def build():
    """Load	the	CSV	and	return	a	wide,	feature-engineered	frame."""
    df = pd.read_csv(OUT / "latency_history.csv", parse_dates=["timestamp"])
    wide = df.pivot(index="timestamp", columns="network",
                    values="baseline_latency_ms").sort_index()
    wide.columns = [f"base_{c}" for c in wide.columns]  # base_3G,	base_4G,	base_5G
    idx = wide.index
    tod = idx.hour + idx.minute / 60
    wide["tod_sin"] = np.sin(2 * np.pi * tod / 24)  # time-of-day	(cyclical)
    wide["tod_cos"] = np.cos(2 * np.pi * tod / 24)
    wide["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)  # day-of-week
    wide["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    return wide


def make_windows(wide):
    """Slide	a	LOOKBACK	window	over	the	series;	target	=	baseline	at	each	horizon."""
    feat_cols = list(wide.columns)
    target_cols = [f"base_{n}" for n in NETWORKS]
    feats = wide[feat_cols].values
    tgt = wide[target_cols].values
    maxh = max(HORIZONS)
    X, Y = [], []
    for i in range(LOOKBACK, len(wide) - maxh):
        X.append(feats[i - LOOKBACK:i])
        #	stack	the	3	networks	for	each	horizon	->	3	x	3	=	9	outputs
        Y.append(np.concatenate([tgt[i + h - 1] for h in HORIZONS]))
    return (np.array(X, np.float32), np.array(Y, np.float32),
            feat_cols, target_cols)


if __name__ == "__main__":
    wide = build()
    X, Y, fc, tc = make_windows(wide)
    n = len(X);
    a, b = int(n * 0.7), int(n * 0.85)
    print("windows:", n, "|	X:", X.shape, "|	Y:", Y.shape)
    print("train	/	val	/	test:", a, b - a, n - b)
    print("features:", fc)
    print("any	NaN:", bool(np.isnan(X).any() or np.isnan(Y).any()))

    # --- ADD THIS CODE TO SAVE THE OUTPUT FILES ---
    # Create the directory if it doesn't exist
    OUT.mkdir(parents=True, exist_ok=True)
    # Save the processed data arrays into your "shared-data" directory
    np.save(OUT / "X.npy", X)
    np.save(OUT / "Y.npy", Y)
    print(f"Successfully saved output files to: {OUT.resolve()}")
