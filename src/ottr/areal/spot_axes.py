"""Spot-level areal axes without pseudobulk averaging (step 07).  python -m ottr.areal.spot_axes "<label>:<embedding>,..." [--tag _x]

Model per spot i (spots with a layer and a depth):  z_i = f(depth_i) + theta c_s(i) + sum_m a_m(i) g_m + eps
  f      : layer effect: 'layer' = mean latent of the spot's layer; 'depth' = cubic spline of the continuous Laplace depth
           (no layer identities); 'layer+depth' = layer mean plus a spline of the depth inside each layer
  theta  : technical shift per section, c_s = centred log median UMI of the section (one vector for all spots)
  a, g   : PCA of the residual (5 axes; axes fitted on every 7th spot)
Region labels are used only to read the axes: variance share, layer / region-in-layer / section eta2, trend R2 with the
section number, |r| of region means with the region SA index, stability = |cos| of g fitted on odd vs even sections.
-> tables/spot_axes_<ds><tag>.csv, work/axes/spot_axes_scores<tag>_<variant>_<model>.npy (scores of the used spots)"""
import os, argparse
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import SplineTransformer
from ..config import cfg
from .multi_axes import axes_dir

K = 5; LAY = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']


def eta(v, g):
    s = pd.Series(v); return float(((s.groupby(g).transform('mean') - s.mean()) ** 2).sum() / ((s - s.mean()) ** 2).sum())


def run(items, ds=None, tag=''):
    c = cfg(); ds = ds or c.series['name']
    o = ad.read_h5ad(c.inputs(ds), backed='r').obs
    sec = o.slice.astype(str).values; lay = o.layer.astype(str).values; reg = o.region.astype(str).values
    u = o.depth.values.astype(float); lib = o.library_size.values.astype(float); num = np.array([int(s[1:]) for s in sec])
    ok = np.isin(lay, LAY) & np.isfinite(u); RG = ~np.isin(reg, ['nan', 'None', 'remove', 'unassigned'])
    SEC = sorted(set(sec)); cs = {s: np.log(np.median(lib[sec == s])) for s in SEC}; cm = np.mean(list(cs.values())); ct = np.array([cs[s] - cm for s in sec])
    C = pd.read_csv(c.columns(ds)); sa_reg = C[C.Ridge_SA_index.notna()].groupby('area_level3').Ridge_SA_index.mean()
    odd = np.array([n % 2 == 1 for n in num])
    sec, lay, reg, u, num, ct, odd, RG = (x[ok] for x in (sec, lay, reg, u, num, ct, odd, RG))

    def residual(Z, variant):
        R = Z.copy()
        if variant in ('layer', 'layer+depth'):
            for l in LAY:
                q = lay == l; R[q] -= R[q].mean(0)
        if variant == 'layer+depth':
            for l in LAY:
                q = lay == l
                if q.sum() < 100: continue
                Bs = SplineTransformer(n_knots=5, degree=3).fit_transform(u[q][:, None]); Bs = np.c_[Bs, np.ones(q.sum())]
                beta, *_ = np.linalg.lstsq(Bs, R[q], rcond=None); R[q] -= Bs @ beta
        elif variant == 'depth':
            Bs = SplineTransformer(n_knots=10, degree=3).fit_transform(u[:, None]); Bs = np.c_[Bs, np.ones(len(u))]
            beta, *_ = np.linalg.lstsq(Bs, R, rcond=None); R -= Bs @ beta
        th = (ct @ (R - R.mean(0))) / (ct @ ct); R -= ct[:, None] * th[None, :]
        return R - R.mean(0)

    def pca(R, k=K):
        U, S, Vt = np.linalg.svd(R[::7] if len(R) > 100000 else R, full_matrices=False)
        G = Vt[:k]; tot = (R ** 2).sum(); A = R @ G.T; return A, G, (A ** 2).sum(0) / tot

    rows = []
    for nm, emb in items:
        Z = np.load(c.emb(ds, emb)).astype(np.float64)[ok]; Z = Z[:, Z.std(0) > 0]; Z = Z / Z.std(0)
        for variant in ('layer', 'depth', 'layer+depth'):
            R = residual(Z, variant); A, G, ve = pca(R)
            _, Go, _ = pca(R[odd]); _, Ge, _ = pca(R[~odd])
            np.save(os.path.join(axes_dir(), f'spot_axes_scores{tag}_{variant}_{nm.replace(" ", "_")}.npy'), A.astype(np.float32))
            for m in range(K):
                a = A[:, m]; b = np.polyfit(num, a, 1); tr = 1 - ((a - np.polyval(b, num)) ** 2).sum() / ((a - a.mean()) ** 2).sum()
                rm = pd.Series(a[RG]).groupby(reg[RG]).mean(); sar = sa_reg.reindex(rm.index); q = sar.notna()
                rows.append(dict(model=nm, variant=variant, component=m + 1, var_share=ve[m], layer_eta2=eta(a, lay),
                                 region_eta2_in_layer=np.mean([eta(a[(lay == l) & RG], reg[(lay == l) & RG]) for l in ['L2', 'L3', 'L5', 'L6']]),
                                 section_eta2=eta(a, sec), trend_R2=tr, SA_r_region_means=abs(np.corrcoef(rm[q], sar[q])[0, 1]),
                                 axis_stability_odd_even=abs(Go[m] @ Ge[m])))
            print(nm, variant, 'done', flush=True)
    R = pd.DataFrame(rows); R.to_csv(c.tables(f'spot_axes_{ds}{tag}.csv'), index=False)
    return R


if __name__ == '__main__':
    P = argparse.ArgumentParser(); P.add_argument('items'); P.add_argument('--tag', default=''); a = P.parse_args()
    run([tuple(x.split(':')[:2]) for x in a.items.split(',')], tag=a.tag)
