#	ml/simulate.py
"""Phase	1:	generate	a	synthetic	latency	history	for	3G/4G/5G."""
import numpy as np
import pandas as pd
from pathlib import Path
from common import (NETWORKS, BASELINE_MEAN, DAILY_AMPLITUDE, NOISE_STD,CAPACITY, STEP_MINUTES, load_penalty)

OUT = Path("shared-data");
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)  # fixed	seed	=	reproducible	data
DAYS = 30
steps_per_day = 24 * 60 // STEP_MINUTES
n_steps = DAYS * steps_per_day
index = pd.date_range("2025-01-01	00:00", periods=n_steps,
                      freq=f"{STEP_MINUTES}min")


def daily_shape(ts) -> float:
    """Two	daily	congestion	humps	(mid-morning	and	evening)."""
    h = ts.hour + ts.minute / 60
    return 0.6 * np.sin((h - 6) / 24 * 2 * np.pi) + \
        0.4 * np.sin((h - 18) / 12 * 2 * np.pi)


rows = []
for net in NETWORKS:
    base = np.full(n_steps, BASELINE_MEAN[net], float)
    shape = np.array([daily_shape(t) for t in index])
    base = base + DAILY_AMPLITUDE[net] * shape
    #	random	congestion	spikes,	decaying	over	a	few	steps	(Poisson	events)
    for _ in range(RNG.poisson(DAYS * 1.5)):
        s = RNG.integers(0, n_steps)
        dur = RNG.integers(2, 8)
        mag = RNG.uniform(1.5, 3.0) * DAILY_AMPLITUDE[net]
        for j in range(s, min(s + dur, n_steps)):
            base[j] += mag * np.exp(-(j - s) / dur)
    base = np.clip(base + RNG.normal(0, NOISE_STD[net], n_steps), 1, None)
    #	an	ambient	device	load,	used	only	to	produce	an	observed	latency	column
    kseries = RNG.integers(5, CAPACITY[net] // 2, n_steps)
    observed = base + np.array([load_penalty(net, int(k)) for k in kseries])
    for i in range(n_steps):
        rows.append((index[i], net, int(kseries[i]),
                     round(float(base[i]), 2), round(float(observed[i]), 2)))
df = pd.DataFrame(rows, columns=["timestamp", "network", "device_count",
                                 "baseline_latency_ms", "observed_latency_ms"])
df = df.sort_values(["timestamp", "network"]).reset_index(drop=True)
df.to_csv(OUT / "latency_history.csv", index=False)
print(f"Wrote	{len(df)}	rows	({n_steps}	steps	x	{len(NETWORKS)}	networks)")
print(df.groupby("network")[["baseline_latency_ms", "observed_latency_ms"]].mean().round(1))
