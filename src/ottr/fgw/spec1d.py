"""FGW 1D anchor spec: nodes = cortical columns (step 03).  python -m ottr.fgw.spec1d --ds pfcserL --alpha 0.7 --circ --pairs_k 2

For every section: ordered complete columns, the spot ids of every (column, layer) block, the arc-length coordinate.
For every section pair: balanced fused Gromov-Wasserstein coupling T between the columns of A and B,
    min_T  (1 - alpha) sum_ij M_ij T_ij  +  alpha sum_ijkl (CA_ik - CB_jl)^2 T_ij T_kl,   T 1 = p, T' 1 = q (uniform)
    M  = squared distance of the column profiles (smoothed PCA, z-scored within section), scaled to max 1
    CA = |arc_i - arc_k| along the cortex (circular for ring sections with --circ), scaled to max 1
alpha weights the structure (keep neighbouring columns neighbours) against the feature match. The model's training loss pulls
the latent of every (column, layer) block of A towards the blocks of B that T assigns to it (ottr.inspire.model).
-> anchors/fgw_spec_<ds><tag>.pkl"""
import os, pickle, argparse
import numpy as np
import ot
from ..config import cfg, ensure
from .common import Data, column_blocks, profiles, local_index, is_ring, default_space, LAYERS


def build(ds, alpha=0.7, circ=True, pairs_k=2, space=None, tag=None):
    D = Data(ds, pairs_k); space = space or default_space(ds)
    full = D.obs.slice_full.astype(str).values; ids = D.obs.index.astype(str).values
    spec = {'ds': ds, 'alpha': alpha, 'circular': circ, 'space': space, 'layers': LAYERS, 'slices': {}, 'pairs': []}
    cache = {}
    for s in D.slices:
        cols, blocks, arc, area = column_blocks(D, s); sf = full[D.sl == s][0]
        if len(cols) == 0: continue                 # sections without complete columns (pole tip) get no anchors
        spec['slices'][sf] = dict(short=s, cols=cols, arc=arc, area2=area, blocks=[{l: ids[v] for l, v in bl.items()} for bl in blocks])
        cache[s] = (cols, blocks, arc, area)

    def struct(s, arc, cols):
        d = np.abs(arc[:, None] - arc[None, :])
        if circ:
            ring, L = is_ring(D, s, cols, arc)
            if ring: d = np.minimum(d, L - d)
        return d / d.max()

    for sa, sb in D.pairs():
        ia, ib = np.flatnonzero(D.sl == sa), np.flatnonzero(D.sl == sb); pa, pb = local_index(ia), local_index(ib)
        ZA, ZB = D.space(space, ia, ib); (_, blA, arcA, arA), (_, blB, arcB, arB) = cache[sa], cache[sb]
        PA, PB = profiles(ZA, blA, pa), profiles(ZB, blB, pb)
        M = ot.dist(PA, PB); M /= M.max()
        CA = struct(sa, arcA, cache[sa][0]); CB = struct(sb, arcB, cache[sb][0]); p, q = ot.unif(len(PA)), ot.unif(len(PB))
        T = np.asarray(ot.gromov.fused_gromov_wasserstein(M, CA, CB, p, q, loss_fun='square_loss', alpha=alpha))
        j = T.argmax(1); ok = (arA != 'nan') & (arB[j] != 'nan'); acc = float(np.mean(arA[ok] == arB[j][ok]))
        print(ds, sa, sb, 'columns', T.shape, 'argmax area precision', round(acc, 3), flush=True)
        spec['pairs'].append(dict(a=full[D.sl == sa][0], b=full[D.sl == sb][0], T=T.astype(np.float32), CA=CA, CB=CB, p=p, q=q, argmax_area2_precision=acc))
    tag = tag if tag is not None else f"_a{str(alpha).replace('.', '')}{'c' if circ else ''}_k{pairs_k}"
    out = os.path.join(ensure(cfg().anchors), f'fgw_spec_{ds}{tag}.pkl'); pickle.dump(spec, open(out, 'wb')); print('saved', out)
    return out


if __name__ == '__main__':
    P = argparse.ArgumentParser(); P.add_argument('--ds', default=cfg().series['name']); P.add_argument('--alpha', type=float, default=0.7)
    P.add_argument('--circ', action='store_true'); P.add_argument('--pairs_k', type=int, default=2); P.add_argument('--space', default=None)
    a = P.parse_args(); build(a.ds, a.alpha, a.circ, a.pairs_k, a.space)
