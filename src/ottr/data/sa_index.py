"""Sensory-association (SA) index of every cortical column from the frozen adult SA classifier.

The classifier (external, paths.sa_model: 09_model.pkl, 08_calibration.json) was trained on annotated adult columns. Per column:
unweighted mean over the available L2-L6 of the layer-block expression of its 476 genes; gene-wise z-score within the chip
(ddof 1, upper clip 10); logistic-regression margin times its orientation; calibrated to (margin - midpoint) / half range.
Block expression in 'pseudobulk' mode = log1p(1e4 x mean counts / mean library) of the block's spots."""
import json, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from ..config import cfg

LAYERS = ['L2', 'L3', 'L4', 'L5', 'L6']


def lognorm_blocks(Xc, lib, mode='pseudobulk'):
    """Returns f(spot indices, gene indices) -> block expression. 'spot': mean of per-spot log1p CP10k;
    'pseudobulk': log1p(1e4 * mean x / mean library)."""
    def f(idx, gi):
        x = Xc[idx][:, gi]; x = x.toarray() if sp.issparse(x) else np.asarray(x)
        if mode == 'spot': return np.log1p(1e4 * x / lib[idx][:, None]).mean(0)
        return np.log1p(1e4 * x.mean(0) / lib[idx].mean())
    return f


def sa_index(lognorm_fn, cols_of, genes_all):
    """cols_of: {(section, column): {layer: spot indices}} of ONE chip; genes_all: gene names of the chip's matrix.
    Returns {(section, column): calibrated SA index}."""
    d = cfg().path('sa_model')
    M = pickle.load(open(f'{d}/09_model.pkl', 'rb')); genes = list(M['genes']); cal = json.load(open(f'{d}/08_calibration.json'))
    gi = pd.Index(genes_all).get_indexer(genes); assert (gi >= 0).all(), 'SA genes missing'
    rows, vals = [], []
    for key, blocks in cols_of.items():
        prof = [lognorm_fn(idx, gi) for l, idx in blocks.items() if l in LAYERS and len(idx)]
        if not prof: continue
        rows.append(key); vals.append(np.mean(prof, axis=0))
    V = np.array(vals); ch = np.array([k[0] for k in rows]); Z = np.zeros_like(V)
    for c in np.unique(ch):
        m = ch == c; mu = V[m].mean(0); sd = V[m].std(0, ddof=1); ok = sd > 1e-10
        z = np.zeros_like(V[m]); z[:, ok] = (V[m][:, ok] - mu[ok]) / sd[ok]; z[z > 10] = 10; Z[m] = z
    margin = M['model'].decision_function(Z) * M['ridge_orientation']
    return dict(zip(rows, (margin - cal['calibration_midpoint']) / cal['calibration_half_range']))
