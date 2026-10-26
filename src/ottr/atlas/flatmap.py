"""Our spots and axes on the atlas flat map (step 08).

assign()   python -m ottr.atlas.flatmap assign
           Place the series spots (bin100 grid) into the atlas segments (bin50 mask of T408-T428, paths.atlas_mask): a spot covers
           the 2 x 2 bin50 pixels (X/50 + dx, Y/50 + dy), dx, dy in {0, 1}, the global offset chosen by the best agreement of
           our region label with the segment's atlas region; segment id = majority over the covered pixels.
           -> tables/atlas/spot2segment.csv
project()  python -m ottr.atlas.flatmap project --tag <multi_axes tag> --models "FGW 2D" --emb <embedding> --spot_tag <spot_axes tag> --out <name>
           Per spot and model: areal axes 1-5 ((z - layer mean) . pb_G[m, layer], the pseudobulk axis evaluated on single spots),
           spot PCs, column PCs of the spot's column, spot-level model scores (ottr.areal.spot_axes, variant layer+depth).
           Per segment: mean over its spots (>= 5); smoothed copy as Huang et al. smooth_kNN (mean of the 25 nearest segments on
           (AP, 3.5 x Y), 10 rounds). Each component's sign is flipped so that its Spearman with its best-matching atlas PC is
           positive. -> tables/atlas/<out>/segment_values.csv, segment_matches.csv"""
import os, argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from ..config import cfg, ensure
from ..areal.multi_axes import axes_dir

LAYERS = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']; MIN = 5


def assign():
    c = cfg(); OUT = ensure(c.tables('atlas'))
    K = pd.read_csv(c.path('atlas_mask')); S = pd.read_csv(f"{c.path('atlas_segments')}/segments_meta.csv", index_col=0)
    o = ad.read_h5ad(c.inputs(c.series['name']), backed='r'); obs = o.obs; xy = np.asarray(o.obsm['spatial'], float)
    sec = obs.slice.astype(str).values; lay = obs.layer.astype(str).values; reg = obs.region.astype(str).values
    num = np.array([int(s[1:]) for s in sec]); W = (num >= 408) & (num <= 428)
    look = {(s, x, y): i for s, x, y, i in zip(K.slice, K.x, K.y, K.id)}

    def at(off):
        ids = []
        for s, X, Y in zip(sec[W], xy[W, 0], xy[W, 1]):
            bx, by = int(X // 50) + off[0], int(Y // 50) + off[1]
            cc = [look.get((s, bx + a, by + b)) for a in (0, 1) for b in (0, 1)]; cc = [v for v in cc if v is not None]
            ids.append(max(set(cc), key=cc.count) if cc else None)
        return np.array(ids, dtype=object)

    best = None
    for off in [(0, 0), (-1, -1), (-1, 0), (0, -1)]:
        ids = at(off); got = np.array([v is not None for v in ids])
        sr = np.array([S.region.get(v, 'nan') if v is not None else 'nan' for v in ids]); ok = got & ~np.isin(reg[W], ['nan', 'None', 'remove', 'unassigned'])
        agree = (sr[ok] == reg[W][ok]).mean(); print(f'offset {off}: region agreement {agree:.3f}', flush=True)
        if best is None or agree > best[0]: best = (agree, off, ids)
    agree, off, ids = best; print('chosen offset', off)
    D = pd.DataFrame({'spot': obs.index[W], 'slice': sec[W], 'layer': lay[W], 'region': reg[W], 'segment': ids})
    D = D[D.segment.notna()]; D['segment_region'] = S.region.reindex(D.segment).values
    D.to_csv(f'{OUT}/spot2segment.csv', index=False); print('saved', len(D), 'spots')


def project(models, embs, axes_tag, spot_tag, out):
    """models: labels inside the multi_axes run axes_tag; embs: {label: embedding}; spot_tag: tag of the spot_axes run."""
    c = cfg(); ds = c.series['name']; OUTD = ensure(c.tables(f'atlas/{out}'))
    o = ad.read_h5ad(c.inputs(ds), backed='r'); obs = o.obs; dep = obs.depth.values.astype(float); sec = obs.slice.astype(str).values
    lay = obs.layer.astype(str).values; col = obs.col.values.astype(int)
    A = pd.read_csv(c.tables('atlas/spot2segment.csv')).set_index('spot'); pos = pd.Series(np.arange(len(obs)), index=obs.index)
    idx = pos[A.index].values; seg = A.segment.values
    S = pd.read_csv(f"{c.path('atlas_segments')}/segments_meta.csv", index_col=0); P = pd.read_csv(f"{c.path('atlas_segments')}/segments_pca.csv", index_col=0)
    T = pd.read_csv(c.tables(f'axes_{ds}{axes_tag}_comp.csv'))
    SAF = pd.read_csv(c.tables(f'spot_axes_{ds}{spot_tag}.csv')); SAF = SAF[SAF.variant == 'layer+depth']
    segs = sorted(set(seg)); REF = pd.concat([P.loc[segs], S.loc[segs, ['Pr-Al-Index', 'Myelinmap', 'Hierachical']]], axis=1)
    XY = np.c_[S.loc[segs, 'AP'].values, S.loc[segs, 'Y_axis'].values * 3.5]
    nn = NearestNeighbors(n_neighbors=25).fit(XY).kneighbors(XY, return_distance=False)

    def smooth(v, rounds=10):
        r = v.copy(); ok = np.isfinite(r); r[~ok] = np.nanmean(r)
        for _ in range(rounds): r = r[nn].mean(1)
        return r

    vals, rows = {}, []
    for mod in models:
        Z = np.load(os.path.join(axes_dir(), f'{ds}{axes_tag}_{mod.replace(" ", "_")}.npz'), allow_pickle=True)
        E = np.load(c.emb(ds, embs[mod])).astype(np.float64)[:, Z['keepf']]
        lm = {l: E[lay == l].mean(0) for l in LAYERS}; G = Z['pb_G']; sp = {}
        for k in range(G.shape[0]):
            v = np.full(len(idx), np.nan)
            for j, l in enumerate(LAYERS):
                q = lay[idx] == l
                if q.any(): v[q] = (E[idx[q]] - lm[l]) @ G[k, j]
            sp[('areal', k + 1)] = v
        for k in range(5): sp[('spot', k + 1)] = Z['spot_scores'][idx, k].astype(float)
        cd = dict(zip(Z['col_keys'], Z['col_scores'].astype(float)))
        for k in range(5): sp[('column', k + 1)] = np.array([cd[f'{s}|{cc}'][k] if f'{s}|{cc}' in cd else np.nan for s, cc in zip(sec[idx], col[idx])])
        okm = np.isin(lay, LAYERS) & np.isfinite(dep); okpos = np.cumsum(okm) - 1
        SA = np.load(os.path.join(axes_dir(), f'spot_axes_scores{spot_tag}_layer+depth_{mod.replace(" ", "_")}.npy'))
        for k in range(5): sp[('spotmodel', k + 1)] = np.where(okm[idx], SA[okpos[idx], k], np.nan)
        for (kind, k), v in sp.items():
            d = pd.DataFrame({'s': seg, 'v': v}).dropna(); g = d.groupby('s').v.agg(['mean', 'size']); m = g['mean'].where(g['size'] >= MIN).reindex(segs).values
            cr = {cc: spearmanr(m, REF[cc], nan_policy='omit')[0] for cc in REF.columns}
            best = max([cc for cc in REF.columns if cc.startswith('PC_')], key=lambda cc: abs(cr[cc])); sgn = 1 if cr[best] >= 0 else -1
            m = sgn * m; name = f'{mod}|{kind}|{k}'; vals[name] = m; vals[name + '_smooth'] = smooth(m)
            if kind == 'spotmodel':
                f = SAF[(SAF.model == mod) & (SAF.component == k)].iloc[0]
                rd = 'SA axis' if (f.SA_r_region_means >= 0.6 and f.region_eta2_in_layer >= 0.3) else 'region axis' if f.region_eta2_in_layer >= 0.3 else 'within-layer variation'
                r = pd.DataFrame([dict(var_share=f.var_share, reading=f'{rd} (region η² {f.region_eta2_in_layer:.2f})', splithalf=f.axis_stability_odd_even)])
            else:
                an = {'areal': 'pseudobulk', 'spot': 'spot', 'column': 'column'}[kind]; r = T[(T.model == mod) & (T.analysis == an) & (T.comp == k)]
            rows.append(dict(model=mod, kind=kind, comp=k, var_share=r.var_share.iloc[0] if len(r) else np.nan, reading=r.reading.iloc[0] if len(r) else '',
                             splithalf=r.splithalf.iloc[0] if (len(r) and 'splithalf' in r) else np.nan, n_segments=int(np.isfinite(m).sum()),
                             best_atlas=best, best_rho=abs(cr[best]), **{f'rho_{cc}': sgn * cr[cc] for cc in REF.columns}))
        print(mod, 'done', flush=True)
    V = pd.DataFrame(vals, index=segs); V.index.name = 'segment'; V.to_csv(f'{OUTD}/segment_values.csv')
    R = pd.DataFrame(rows); R.to_csv(f'{OUTD}/segment_matches.csv', index=False)
    return V, R


if __name__ == '__main__':
    P_ = argparse.ArgumentParser(); P_.add_argument('what', choices=['assign', 'project']); P_.add_argument('--tag', default='')
    P_.add_argument('--models', default='FGW 2D'); P_.add_argument('--embs', default=''); P_.add_argument('--spot_tag', default=''); P_.add_argument('--out', default='flatmap')
    a = P_.parse_args()
    if a.what == 'assign': assign()
    else:
        models = a.models.split(','); project(models, dict(zip(models, a.embs.split(','))), a.tag, a.spot_tag, a.out)
