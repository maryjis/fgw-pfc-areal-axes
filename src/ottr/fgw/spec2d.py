"""FGW 2D anchor spec: nodes = (column x depth bin) blocks (step 03).  python -m ottr.fgw.spec2d --ds pfcserL --alpha 0.7 --circ --pairs_k 2

Why 2D: a 1D coupling of whole columns cannot tell whether superficial and deep parts of a column match; with nodes split
into 8 depth bins (relative depth from the manual layers) the structure term also keeps depth order, and the coupling
aligns the cortex as a 2D sheet (along the cortex x through the layers) without being told the layers.
  node features: mean smoothed-PCA of the node's spots, z-scored within the section
  structure    : 2D distance sqrt(dx^2 + dz^2) in mm (dx = arc along the cortex x 0.5 um / unit, dz = depth-bin difference
                 x 2 mm cortical thickness / 8 bins), circular along the cortex for ring sections, scaled to max 1
  coupling     : balanced FGW, alpha 0.7 (as FGW 1D)
The spec has one pseudo-layer 'all', so the model's FGW code treats every 2D block as one node.
-> anchors/fgw2d_spec_<ds><tag>.pkl"""
import os, pickle, argparse, warnings
import numpy as np
import pandas as pd
import ot
from ..config import cfg, ensure
from .common import Data, column_blocks, profiles, local_index, is_ring, default_space

NBIN = 8; THICK_MM = 2.0; UM_PER_UNIT = 0.5


def slice_nodes(D, s, Z, pos, circ=False):
    """2D nodes of section s: features F, structure C, node column k, depth bin b, majority layer, spot indices."""
    cols, blocks, arc, area = column_blocks(D, s); colidx = {cc: k for k, cc in enumerate(cols)}
    m = np.flatnonzero((D.sl == s) & (D.col >= 0) & np.isfinite(D.depth)); m = m[np.isin(D.col[m], cols)]
    b = np.clip((D.depth[m] * NBIN).astype(int), 0, NBIN - 1)
    df = pd.DataFrame({'g': m, 'k': [colidx[cc] for cc in D.col[m]], 'b': b, 'lay': D.layer[m]})
    nodes = []
    for (k, bb), gg in df.groupby(['k', 'b']):
        if len(gg) < 3: continue
        nodes.append(dict(k=k, b=bb, idx=gg.g.values, lay=gg.lay.value_counts().index[0]))
    F = np.array([Z[pos(n['idx'])].mean(0) for n in nodes]); F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    k = np.array([n['k'] for n in nodes]); bb = np.array([n['b'] for n in nodes])
    dx = np.abs(arc[k][:, None] - arc[k][None]); dz = np.abs(bb[:, None] - bb[None]) * THICK_MM / NBIN
    if circ:
        ring, Lr = is_ring(D, s, cols, arc)
        if ring: dx = np.minimum(dx, Lr - dx)
    C = np.sqrt((dx * UM_PER_UNIT / 1000) ** 2 + dz ** 2); C /= C.max()
    return dict(F=F, C=C, k=k, b=bb, lay=np.array([n['lay'] for n in nodes]), area=area, ncol=len(cols), node_idx=[n['idx'] for n in nodes],
                cols=cols, col_mean=profiles(Z, blocks, pos))


def build(ds, alpha=0.7, circ=True, pairs_k=2, space=None, tag=None):
    warnings.filterwarnings('ignore')
    D = Data(ds, pairs_k); space = space or default_space(ds)
    full = D.obs.slice_full.astype(str).values; ids = D.obs.index.astype(str).values
    spec = {'ds': ds, 'alpha': alpha, 'circular': circ, 'space': space, 'layers': ['all'], 'slices': {}, 'pairs': [], 'type': 'blocks2d'}
    for sa, sb in D.pairs():
        ia, ib = np.flatnonzero(D.sl == sa), np.flatnonzero(D.sl == sb); ZA, ZB = D.space(space, ia, ib)
        NA, NB = slice_nodes(D, sa, ZA, local_index(ia), circ), slice_nodes(D, sb, ZB, local_index(ib), circ)
        for s, N in ((sa, NA), (sb, NB)):
            sf = full[D.sl == s][0]
            if sf not in spec['slices']:
                spec['slices'][sf] = dict(short=s, cols=np.arange(len(N['k'])), arc=np.arange(len(N['k']), dtype=float), area2=N['area'][N['k']],
                                          blocks=[{'all': ids[v]} for v in N['node_idx']], node_col=N['k'], node_bin=N['b'], node_layer=N['lay'])
        M = ot.dist(NA['F'], NB['F']); M /= M.max(); p, q = ot.unif(len(NA['C'])), ot.unif(len(NB['C']))
        T = np.asarray(ot.gromov.fused_gromov_wasserstein(M, NA['C'], NB['C'], p, q, loss_fun='square_loss', alpha=alpha))
        same_area = float((T * (NA['area'][NA['k']][:, None] == NB['area'][NB['k']][None, :])).sum() / T.sum())
        same_layer = float((T * (NA['lay'][:, None] == NB['lay'][None, :])).sum() / T.sum())
        print(ds, sa, sb, 'nodes', T.shape, 'mass same area %.3f same layer %.3f' % (same_area, same_layer), flush=True)
        spec['pairs'].append(dict(a=full[D.sl == sa][0], b=full[D.sl == sb][0], T=T.astype(np.float32), CA=NA['C'], CB=NB['C'], p=p, q=q,
                                  argmax_area2_precision=same_area))
    tag = tag if tag is not None else f"_a{str(alpha).replace('.', '')}{'c' if circ else ''}_k{pairs_k}"
    out = os.path.join(ensure(cfg().anchors), f'fgw2d_spec_{ds}{tag}.pkl'); pickle.dump(spec, open(out, 'wb'))
    print('saved', out, len(spec['slices']), 'slices', len(spec['pairs']), 'pairs')
    return out


if __name__ == '__main__':
    P = argparse.ArgumentParser(); P.add_argument('--ds', default=cfg().series['name']); P.add_argument('--alpha', type=float, default=0.7)
    P.add_argument('--circ', action='store_true'); P.add_argument('--pairs_k', type=int, default=2); P.add_argument('--space', default=None)
    a = P.parse_args(); build(a.ds, a.alpha, a.circ, a.pairs_k, a.space)
