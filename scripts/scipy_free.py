"""Tiny rank-correlation helper (avoids a scipy dependency)."""
import numpy as np


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    if len(x) < 3:
        return float('nan')
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])
