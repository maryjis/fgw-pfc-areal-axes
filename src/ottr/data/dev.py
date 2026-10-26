"""Inputs of one developmental age group (P0, P32, 3M, 7M) (step 01).  python -m ottr.data.dev --ds dev7M --stage 7M

As ottr.data.series, plus: the annotation of the young brains is joined on the spot coordinates (y mirrored) from
<sample>_b100_anno.csv (manual_layer -> layer, area -> region: mPFC / oPFC / vlPFC / dlPFC, pseudoaxis) and
<sample>_b100_cdepth-sa.csv (sa_loess); the genes are the ADULT model genes in the adult order (missing genes = zero columns),
so the adult encoder can be applied; per column also mean sa_loess and pseudoaxis; pseudo-regions obs.preg = area x tertile
of the pseudoaxis within the area (for region-level summaries).
Requires inputs/<series>.h5ad (the adult gene list)."""
import os, glob, argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from ..config import cfg, ensure
from ..utils.columns import build_laplace
from ..utils.labels import DEV_AREAS
from .sa_index import sa_index, lognorm_blocks, LAYERS
from .series import qc


def prepare(ds, stage):
    c = cfg(); P = c['paths']
    genes = list(ad.read_h5ad(c.inputs(c.series['name']), backed='r').var_names)
    ind = ensure(c.run_input(ds)); ensure(os.path.dirname(c.inputs(ds)))
    for p in sorted(glob.glob(f"{c.path('raw_dev')}/*_{stage}_PFC_Sub-CTX_b100_filtered.h5ad")):
        l = f'{ind}/{os.path.basename(p)}'
        if not os.path.lexists(l): os.symlink(p, l)
    files = sorted(glob.glob(f'{ind}/*_b100_filtered.h5ad'))
    obs, Xs, xys, cols_of, raw_for_sa = [], [], [], {}, {}
    for i, p in enumerate(files):
        a = qc(p); sid = os.path.basename(p).replace('_b100_filtered.h5ad', ''); t = sid.split('_')[0]
        xy = np.asarray(a.obsm['spatial'], float); k = list(zip(np.round(xy[:, 0]).astype(int), np.round(xy[:, 1]).astype(int)))
        an = pd.read_csv(f"{c.path('dev_annotation')}/{sid}_b100_anno.csv"); cd = pd.read_csv(f"{c.path('cdepth_sa')}/{sid}_b100_cdepth-sa.csv")
        kk = lambda d: list(zip(d.spatial_1.astype(int), (-d.spatial_2).astype(int)))   # annotation tables have y mirrored
        an.index = kk(an); cd.index = kk(cd); an = an[~an.index.duplicated()]; cd = cd[~cd.index.duplicated()]
        lay = an.manual_layer.reindex(k).astype(str).values; area = an.area.reindex(k).astype(str).values
        pax = an.pseudoaxis.reindex(k).values.astype(float); sal = cd.sa_loess.reindex(k).values.astype(float)
        print(sid, 'annotated', np.mean(lay != 'nan').round(3), pd.Series(area).value_counts().to_dict(), flush=True)
        lib = np.asarray(a.X.sum(1)).ravel().astype(np.float32); lib[lib <= 0] = 1
        try: col, s, d = build_laplace(xy, lay)
        except Exception as e: print(sid, 'no columns:', repr(e)[:120]); col = np.full(len(lay), -1); s = d = np.full(len(lay), np.nan)
        gi = pd.Index(a.var_names); X = sp.csr_matrix(a.X).astype(np.float32); pos = gi.get_indexer(genes)
        Xg = sp.lil_matrix((a.n_obs, len(genes)), dtype=np.float32); ok = pos >= 0
        Xg[:, np.flatnonzero(ok)] = X[:, pos[ok]]; print(sid, 'adult genes missing', int((~ok).sum()), flush=True)
        Xs.append(Xg.tocsr()); xys.append(xy)
        obs.append(pd.DataFrame({'spot_id': [f'{b}-{i}' for b in a.obs_names.astype(str)], 'slice': t, 'slice_full': sid, 'group': 'PFC', 'animal': sid.split('_')[1],
                                 'region': area, 'layer': lay, 'library_size': lib, 'col': col, 'arc': s, 'depth': d, 'pseudoaxis': pax, 'sa_loess': sal}))
        fn = lognorm_blocks(X, lib, 'pseudobulk')
        for cc in np.unique(col[col >= 0]):
            m = col == cc; cols_of[(t, int(cc))] = {l: np.flatnonzero(m & (lay == l)) for l in LAYERS}
        raw_for_sa[t] = (fn, list(a.var_names))
    O = pd.concat(obs, ignore_index=True)
    O['preg'] = 'nan'
    for (t, ar), g in O[O.region.isin(DEV_AREAS) & O.pseudoaxis.notna()].groupby(['slice', 'region']):
        q = pd.qcut(g.pseudoaxis.rank(method='first'), 3, labels=['a', 'b', 'c']); O.loc[g.index, 'preg'] = ar + '_' + q.astype(str)
    O = O.set_index('spot_id', drop=False)
    A = ad.AnnData(sp.vstack(Xs).tocsr(), obs=O, var=pd.DataFrame(index=genes)); A.obsm['spatial'] = np.vstack(xys).astype(np.float32)
    A.write_h5ad(c.inputs(ds), compression='gzip')
    sai = {}
    for t, (fn, genes_all) in raw_for_sa.items():
        co = {k: v for k, v in cols_of.items() if k[0] == t}
        if co: sai.update(sa_index(fn, co, genes_all))
    rows = []
    for (t, cc), bl in cols_of.items():
        m = (O.slice.values == t) & (O.col.values == cc); r = O.region.values[m]; r = r[np.isin(r, DEV_AREAS)]
        lev = pd.Series(r).value_counts().index[0] if len(r) else 'nan'
        rows.append(dict(slice=t, col=cc, area_level3=lev, mean_x=A.obsm['spatial'][m, 0].mean(), mean_y=A.obsm['spatial'][m, 1].mean(), area2=lev, SA='unassigned',
                         Ridge_SA_index=sai.get((t, cc), np.nan), sa_loess=np.nanmean(O.sa_loess.values[m]), pseudoaxis=np.nanmean(O.pseudoaxis.values[m])))
    C = pd.DataFrame(rows).sort_values(['slice', 'col']); C.to_csv(c.columns(ds), index=False)
    print(ds, 'spots', A.n_obs, 'columns', len(C))


if __name__ == '__main__':
    P = argparse.ArgumentParser(); P.add_argument('--ds', required=True); P.add_argument('--stage', required=True); a = P.parse_args()
    prepare(a.ds, a.stage)
