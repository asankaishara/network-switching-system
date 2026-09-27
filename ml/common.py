import numpy as np

#	ml/common.py
"""Shared	constants	and	the	network	load	model."""
NETWORKS = ["3G", "4G", "5G"]

#	Ambient	(load-independent)	mean	latency	in	milliseconds.
BASELINE_MEAN = {"3G": 120.0, "4G": 50.0, "5G": 12.0}

#	Amplitude	of	the	daily	congestion	swing	added	on	top	of	the	mean.
DAILY_AMPLITUDE = {"3G": 25.0, "4G": 15.0, "5G": 6.0}

#	Standard	deviation	of	observation	noise.
NOISE_STD = {"3G": 6.0, "4G": 3.0, "5G": 1.5}

#	Load	model:	extra	latency	for	k	devices	=	ALPHA	*	k**BETA.
#	5G	degrades	fastest	as	it	fills	up	(small	effective	capacity);
#	3G	degrades	slowly	but	already	starts	high.
LOAD_ALPHA = {"3G": 0.45, "4G": 0.55, "5G": 0.90}
LOAD_BETA = {"3G": 1.25, "4G": 1.30, "5G": 1.40}
CAPACITY = {"3G": 100, "4G": 70, "5G": 45}

N_DEVICES = 100

STEP_MINUTES = 10
#	Maximum	devices	each	network	can	hold	(e.g.	5G	coverage	limit).
#	total	devices	to	distribute
#	time	resolution
SWITCH_THRESHOLD = 0.15  # only	switch	if	predicted	improvement	>	15%


def load_penalty(network: str, k: int) -> float:
    """Extra	latency	(ms)	contributed	by	putting	k	devices	on	a	network.
    Convex	and	increasing	in	k,	which	is	what	makes	water-filling	optimal."""
    if k <= 0:
        return 0.0
    return LOAD_ALPHA[network] * (k ** LOAD_BETA[network])



class Scaler:
    """Minimal standardizer — numpy replacement for sklearn StandardScaler."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, A):
        A = np.asarray(A, dtype=np.float64)
        self.mean_ = A.mean(axis=0)
        self.std_ = A.std(axis=0)          # population std (ddof=0), matches sklearn
        self.std_[self.std_ == 0] = 1.0    # guard against divide-by-zero
        return self

    def transform(self, A):
        return (np.asarray(A, dtype=np.float64) - self.mean_) / self.std_

    def inverse_transform(self, A):
        return np.asarray(A, dtype=np.float64) * self.std_ + self.mean_