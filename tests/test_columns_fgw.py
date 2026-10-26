"""Columns on a curved cortical ribbon and FGW couplings on synthetic sections."""
import numpy as np


def ribbon(n_along=120, n_depth=24, radius=4000.0):
    """Quarter ring of grid spots: depth bands as tests/synthetic.py (L1 outside)."""
    from synthetic import layer_of
    pts, lay = [], []
    for i in range(-40, 160):
        for j in range(-40, 160):
            x, y = i * 100.0, j * 100.0; r = np.hypot(x, y); t = np.arctan2(y, x)
            if 0 <= t <= np.pi / 2 and radius <= r < radius + n_depth * 100:
                pts.append((x, y)); lay.append(layer_of(int((radius + n_depth * 100 - r) // 100)))
    return np.array(pts), np.array(lay)


def test_laplace_columns_on_ribbon():
    from ottr.utils.columns import build_laplace
    xy, lay = ribbon(); col, s, u = build_laplace(xy, lay)
    ids = np.unique(col[col >= 0]); mid_len = np.pi / 2 * (6400 - 0.6 * 1900)            # arc at the mid-depth line (r ~ 5260)
    assert 0.6 * mid_len / 405 <= len(ids) <= 1.1 * mid_len / 405                         # ~405 units wide (narrow wedges merged)
    ang = [np.arctan2(*xy[col == c].mean(0)[::-1]) for c in ids]
    assert np.all(np.diff(ang) < 0) or np.all(np.diff(ang) > 0)                           # numbered in order along the cortex
    for c in ids:                                                                         # every column spans the layers
        assert all(((col == c) & (lay == l)).sum() >= 3 for l in ('L2', 'L3', 'L5', 'L6'))
    mu = [np.nanmean(u[lay == l]) for l in ('L1', 'L2', 'L3', 'L4', 'L5', 'L6')]; assert np.all(np.diff(mu) > 0), mu   # Laplace depth follows the layers


def test_fgw2d_coupling_keeps_depth(syn):
    from ottr.fgw import spec2d
    import pickle
    spec = pickle.load(open(spec2d.build(syn['ds'], alpha=0.7, circ=False, pairs_k=1, tag='_t'), 'rb'))
    assert len(spec['pairs']) == 5
    for P in spec['pairs']:
        A, B = spec['slices'][P['a']], spec['slices'][P['b']]; T = P['T']
        same_bin = (T * (A['node_bin'][:, None] == B['node_bin'][None, :])).sum() / T.sum()
        near_col = (T * (np.abs(A['node_col'][:, None] - B['node_col'][None, :]) <= 1)).sum() / T.sum()
        assert same_bin > 0.8 and near_col > 0.8, (same_bin, near_col)                  # identical geometry -> diagonal coupling
