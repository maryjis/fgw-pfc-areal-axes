"""Spatial smoothing of expression on the bin100 grid: mean over a disk of radius r grid steps (1 step = 100 coordinate units
= 50 um), uniform weights over the spots present in the disk (including the spot itself), computed per section by FFT
convolution of the grid (cost independent of r)."""
import numpy as np
from scipy.signal import fftconvolve

STEP = 100.0


def disk(r):
    """0/1 kernel of the grid offsets (dx, dy) with dx^2 + dy^2 <= r^2."""
    k = int(np.floor(r)); dx, dy = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing='ij')
    return (dx ** 2 + dy ** 2 <= r * r + 1e-9).astype(np.float32)


def grid_positions(sl, xy, step=STEP):
    """{section: (spot indices, integer grid positions, grid shape)}."""
    G = {}
    for s in dict.fromkeys(sl):
        idx = np.flatnonzero(sl == s); ij = np.round((xy[idx] - xy[idx].min(0)) / step).astype(int)
        G[s] = (idx, ij, tuple(ij.max(0) + 1))
    return G


def disk_mean(get_rows, n_genes, G, r, chunk=200, out=None, min_count=None):
    """Disk mean of every spot. get_rows(idx) -> dense (len(idx), n_genes) float32 rows of the matrix to smooth.
    min_count: lower bound of the neighbour count (multi_axes uses 1; the smoothing curve uses the exact count).
    Returns the smoothed matrix (n_spots, n_genes) and the mean number of spots per disk."""
    K = disk(r); n = sum(len(v[0]) for v in G.values()); X = np.zeros((n, n_genes), np.float32) if out is None else out; counts = []
    for s, (idx, ij, shape) in G.items():
        mask = np.zeros(shape, np.float32); mask[ij[:, 0], ij[:, 1]] = 1
        cnt = np.rint(fftconvolve(mask, K, mode='same'))[ij[:, 0], ij[:, 1]]
        if min_count is not None: cnt = np.maximum(cnt, min_count)
        counts.append(cnt); Ls = get_rows(idx)
        for g0 in range(0, n_genes, chunk):
            grid = np.zeros(shape + (min(chunk, n_genes - g0),), np.float32)
            grid[ij[:, 0], ij[:, 1]] = Ls[:, g0:g0 + chunk]
            S = fftconvolve(grid, K[:, :, None], mode='same', axes=(0, 1))
            X[idx, g0:g0 + chunk] = S[ij[:, 0], ij[:, 1]] / cnt[:, None]
    return X, float(np.mean(np.concatenate(counts)))
