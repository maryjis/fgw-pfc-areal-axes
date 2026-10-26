"""Adult axes in the developing PFC (step 09): P0, P32, 3M, 7M encoded by the ADULT encoder (no training on the young data;
`ottr.inspire.train --encode_only --init_ckpt <adult model>`), so every age is in the adult latent coordinates.

An `AxesModel` names one adult model: its multi_axes output (npz with col_V, spot_V, pb_G; comp table with the readings),
its adult embedding and the embedding of the young data encoded by it.
Projections of the adult directions (as multi_axes):
  column     : (column profile - adult mean column profile) . col_V   (profile = mean of the L2-L6 layer means)
  spot       : (z - adult mean of L1-L6 spots) . spot_V, averaged over the column's L2-L6 spots
  pseudobulk : (z - adult layer mean) . pb_G[m, layer] per spot (the areal-model score), averaged over the column's spots
Functions:
  project()       per age: column scores (tables/dev<tag>/<ds>_columns.csv) and a summary per axis (<ds>_summary.csv): eta2 of the 4
                  areas, eta2 of the sections, rho with sa_loess, r of the 4 area means with the adult area means, spread, shift
  conservation()  per age and axis: share of the age's variance along the adult direction, best |cos| with the age's own top-5 PCs,
                  r of the 4 area means with adults, eta2 of the areas (conservation.csv, conservation_pairs.csv)
  age_areal()     one number per sample (mean or median of the bin scores over L1-L6), Spearman with the age order; top-30 genes
                  at each end of every axis (correlation over the adult pseudobulks) and their mean z per age
  age_metrics()   robustness of the per-sample statistic (mean, median, 20% trimmed mean, layer-balanced median, column median):
                  Spearman with age and its leave-one-sample-out range
Logic copied unchanged from dev_axes_project.py, dev_conservation.py, dev_age_areal.py, dev_age_metrics.py."""
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.stats import spearmanr, trim_mean
from ..config import cfg, ensure
from ..utils.labels import AREA as FINE2AREA, COARSE_DEV, DEV_AREAS
from .multi_axes import axes_dir

LAYERS = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']; K = 5; AGES = ['P0', 'P32', '3M', '7M', 'adult']; AR4 = DEV_AREAS
COARSE = dict(COARSE_DEV, FPC='nan')
LATENT = 'adult encoder (A)'


@dataclass
class AxesModel:
    key: str            # short key, e.g. 'fgw2d_factors'
    label: str          # model label inside the multi_axes run (e.g. 'FGW 2D')
    axes_tag: str       # multi_axes tag of the run that contains this model ('' or '_poisson' ...)
    adult_emb: str      # embedding name of the adult series
    dev_emb: str        # embedding name of the young data encoded by this model
    tag: str            # output tag (tables/dev<tag>/)

    @property
    def npz(self):
        c = cfg(); return np.load(os.path.join(axes_dir(), f"{c.series['name']}{self.axes_tag}_{self.label.replace(' ', '_')}.npz"), allow_pickle=True)

    @property
    def comp(self):
        c = cfg(); T = pd.read_csv(c.tables(f"axes_{c.series['name']}{self.axes_tag}_comp.csv")); return T[T.model == self.label]

    def out(self, name=''):
        return os.path.join(ensure(cfg().tables(f'dev{self.tag}')), name)


def eta(v, g):
    s = pd.Series(v); g = np.asarray(g); return float(((s.groupby(g).transform('mean') - s.mean()) ** 2).sum() / ((s - s.mean()) ** 2).sum())


def col_profiles(E, sec, col, lay):
    """Column profiles (mean of the L2-L6 layer means, >= 3 layers with >= 3 spots), their keys and spot indices."""
    m = (col >= 0) & np.isin(lay, LAYERS[1:]); g = pd.DataFrame({'s': sec[m], 'c': col[m], 'l': lay[m]}).groupby(['s', 'c', 'l']).indices; im = np.flatnonzero(m)
    acc, ids = {}, {}
    for (s, cc, l), ix in g.items():
        if len(ix) >= 3: acc.setdefault((s, cc), []).append(E[im[ix]].mean(0)); ids.setdefault((s, cc), []).extend(im[ix])
    keys = [k for k, v in acc.items() if len(v) >= 3]
    return keys, np.array([np.mean(acc[k], 0) for k in keys]), [np.array(ids[k]) for k in keys]


def _adult(M):
    c = cfg(); Z = M.npz; kf = Z['keepf']; ds = c.series['name']
    ao = ad.read_h5ad(c.inputs(ds), backed='r').obs; Ea = np.load(c.emb(ds, M.adult_emb)).astype(np.float64)[:, kf]
    return Z, kf, ao, Ea


def project(M, ds):
    """Adult column / spot / areal directions on the columns of one young dataset (tables only; figure: ottr.viz.dev)."""
    c = cfg(); Z, kf, ao, Ea = _adult(M); T = M.comp
    alay = ao.layer.astype(str).values; asec = ao.slice.astype(str).values; acol = ao.col.values.astype(int)
    m_spot = Ea[np.isin(alay, LAYERS)].mean(0); m_lay = {l: Ea[alay == l].mean(0) for l in LAYERS}

    def scores(E, sec, col, lay, m_col=None):
        keys, P, ids = col_profiles(E, sec, col, lay); m_col = P.mean(0) if m_col is None else m_col
        sp_ = (E - m_spot) @ Z['spot_V'][:K].T; pb = np.full((len(E), K), np.nan)
        for j, l in enumerate(LAYERS):
            q = lay == l
            if q.any(): pb[q] = (E[q] - m_lay[l]) @ Z['pb_G'][:K, j].T
        U = {'column': (P - m_col) @ Z['col_V'][:K].T, 'spot': np.array([sp_[i].mean(0) for i in ids]), 'pseudobulk': np.array([np.nanmean(pb[i], 0) for i in ids])}
        return keys, U, m_col

    akeys, AU, m_col = scores(Ea, asec, acol, alay)
    AC = pd.read_csv(c.columns(c.series['name'])); AC['slice'] = AC.slice.astype(str); AC = AC.set_index(['slice', 'col'])
    afine = np.array([AC.area_level3.get(k, 'nan') for k in akeys]); acoarse = np.array([COARSE.get(FINE2AREA.get(f, 'nan'), 'nan') for f in afine])
    go = ad.read_h5ad(c.inputs(ds), backed='r').obs; gsec = go.slice.astype(str).values; gcol = go.col.values.astype(int); glay = go.layer.astype(str).values
    GC = pd.read_csv(c.columns(ds)); GC['slice'] = GC.slice.astype(str); GC = GC.set_index(['slice', 'col'])
    E = np.load(c.emb(ds, M.dev_emb)).astype(np.float64)[:, kf]; keys, U, _ = scores(E, gsec, gcol, glay, m_col)
    car = np.array([GC.area2.get(k, 'nan') for k in keys]); csl = np.array([k[0] for k in keys]); csa = np.array([GC.sa_loess.get(k, np.nan) for k in keys])
    rows = []
    for an in ('column', 'spot', 'pseudobulk'):
        for k in range(K):
            v = U[an][:, k]; q = np.isin(car, AR4) & np.isfinite(v); va = AU[an][:, k]; qa = np.isin(acoarse, AR4); f = q & np.isfinite(csa)
            gm = pd.Series(v[q]).groupby(car[q]).mean().reindex(AR4); am = pd.Series(va[qa]).groupby(acoarse[qa]).mean().reindex(AR4)
            t = T[(T.analysis == an) & (T.comp == k + 1)].iloc[0]
            rows.append(dict(latent=LATENT, analysis=an, comp=k + 1, adult_reading=t.reading, eta2_area_group=eta(v[q], car[q]), eta2_section_group=eta(v[q], csl[q]),
                             rho_sa_loess_group=spearmanr(v[f], csa[f])[0], eta2_area_adult=eta(va[qa], acoarse[qa]), r_area_means=np.corrcoef(gm.values, am.values)[0, 1],
                             spread_ratio=np.nanstd(v) / np.nanstd(va), shift_sd=(np.nanmean(v) - np.nanmean(va)) / np.nanstd(va),
                             **{f'group_{a}': gm[a] for a in AR4}, **{f'adult_{a}': am[a] for a in AR4}))
    S = pd.DataFrame(rows); S.to_csv(M.out(f'{ds}_summary.csv'), index=False)
    C = pd.DataFrame({'latent': LATENT, 'slice': csl, 'col': [k[1] for k in keys], 'area': car, 'sa_loess': csa, **{f'{an}_{k + 1}': U[an][:, k] for an in U for k in range(K)}})
    C.to_csv(M.out(f'{ds}_columns.csv'), index=False)
    # adult region means on the same scale (for the scatter figures)
    AR = {an: pd.DataFrame(AU[an]).groupby(afine).mean() for an in AU}
    for an, d in AR.items(): d[d.index.isin(list(FINE2AREA))].to_csv(M.out(f'adult_regions_{an}.csv'))
    return S


def _load_age(M, age, kf):
    c = cfg(); ds, emb = (c.series['name'], M.adult_emb) if age == 'adult' else (f'dev{age}', M.dev_emb)
    o = ad.read_h5ad(c.inputs(ds), backed='r').obs; E = np.load(c.emb(ds, emb)).astype(np.float64)[:, kf]
    C = pd.read_csv(c.columns(ds)); C['slice'] = C.slice.astype(str); C = C.set_index(['slice', 'col'])
    area = {k: COARSE.get(FINE2AREA.get(v, ''), 'nan') for k, v in C.area_level3.items()} if age == 'adult' else C.area2.to_dict()
    return E, o.slice.astype(str).values, o.col.values.astype(int), o.layer.astype(str).values, area


def conservation(M):
    Z = M.npz; kf = Z['keepf']; T = M.comp; CV = Z['col_V'][:5]; PG = Z['pb_G'][:5]; rows, AM = [], {}
    for age in AGES:
        E, sec, col, lay, area = _load_age(M, age, kf); keys, P, ids = col_profiles(E, sec, col, lay); car = np.array([area.get(k, 'nan') for k in keys]); qa = np.isin(car, AR4)
        Pc = P - P.mean(0); tot = (Pc ** 2).sum() / len(Pc); _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
        ok = np.isin(lay, LAYERS); R = E[ok].copy(); ll = lay[ok]
        for l in LAYERS: R[ll == l] -= R[ll == l].mean(0)
        totR = (R ** 2).sum() / len(R); _, _, VtR = np.linalg.svd(R[::4], full_matrices=False); pbs = np.zeros((len(R), 5))
        for j, l in enumerate(LAYERS):
            q = ll == l; pbs[q] = R[q] @ (PG[:, j] / np.linalg.norm(PG[:, j], axis=1, keepdims=True)).T
        oki = np.flatnonzero(ok); pos = np.full(len(E), -1); pos[oki] = np.arange(len(oki))
        for an in ('column', 'pseudobulk'):
            for k in range(5):
                if an == 'column':
                    v = Pc @ CV[k]; share = v.var() / tot; d = CV[k]; own = np.abs(Vt[:5] @ d)
                else:
                    share = sum(((pbs[ll == l, k]) ** 2).sum() for l in LAYERS) / len(R) / totR
                    d = PG[k].mean(0); d /= np.linalg.norm(d); own = np.abs(VtR[:5] @ d)
                    v = np.array([np.mean(pbs[pos[i[pos[i] >= 0]], k]) if (pos[i] >= 0).any() else np.nan for i in ids])
                am = pd.Series(v[qa]).groupby(car[qa]).mean().reindex(AR4); AM[(age, an, k)] = am.values
                t = T[(T.analysis == an) & (T.comp == k + 1)].iloc[0]
                rows.append(dict(age=age, analysis=an, comp=k + 1, reading=t.reading, adult_var_share_model=t.var_share, share=share,
                                 own_cos=own.max(), own_rank=int(own.argmax()) + 1, eta2_area=eta(v[qa], car[qa]), **{f'mean_{a}': x for a, x in zip(AR4, am.values)}))
        print(M.key, age, 'done', flush=True)
    S = pd.DataFrame(rows)
    S['r_area'] = [np.corrcoef(AM[(r.age, r.analysis, r.comp - 1)], AM[('adult', r.analysis, r.comp - 1)])[0, 1] for r in S.itertuples()]
    S.to_csv(M.out('conservation.csv'), index=False)
    pd.DataFrame([dict(analysis=an, comp=k + 1, a=a, b=b, r=np.corrcoef(AM[(a, an, k)], AM[(b, an, k)])[0, 1]) for an in ('column', 'pseudobulk') for k in range(5)
                  for i, a in enumerate(AGES) for b in AGES[i + 1:]]).to_csv(M.out('conservation_pairs.csv'), index=False)
    return S


def areal_scores(E, lay, m_lay, PG):
    """Areal-model score of every bin: (z - adult layer mean) . pb_G[m, layer]."""
    sc = np.full((len(E), K), np.nan)
    for j, l in enumerate(LAYERS):
        q = lay == l
        if q.any(): sc[q] = (E[q] - m_lay[l]) @ PG[:, j].T
    return sc


def age_areal(M, stat='median'):
    c = cfg(); Z = M.npz; kf = Z['keepf']; PG = Z['pb_G'][:K]; T = M.comp; T = T[T.analysis == 'pseudobulk'].set_index('comp')
    ORD = {a: i for i, a in enumerate(AGES)}; SUF = '' if stat == 'mean' else f'_{stat}'
    SRC = [('adult', c.series['name'], M.adult_emb)] + [(a, f'dev{a}', M.dev_emb) for a in AGES[:-1]]
    m_lay = None; rows, EXP, genes = [], [], None
    for age, ds, emb in SRC:
        A = ad.read_h5ad(c.inputs(ds)); o = A.obs; E = np.load(c.emb(ds, emb)).astype(np.float64)[:, kf]
        lay = o.layer.astype(str).values; sec = o.slice.astype(str).values; reg = o.region.astype(str).values; genes = np.array(A.var_names)
        if age == 'adult': m_lay = {l: E[lay == l].mean(0) for l in LAYERS}
        sc = areal_scores(E, lay, m_lay, PG)
        X = (sp.diags(1e4 / o.library_size.values.astype(float)) @ A.X).tocsr().astype(np.float32); X.data = np.log1p(X.data)
        ok = np.isin(lay, LAYERS)
        for s in dict.fromkeys(sec):
            q = (sec == s) & ok
            rows.append(dict(age=age, sample=s, n_bins=int(q.sum()), **{f'axis{k + 1}': (np.nanmedian if stat == 'median' else np.nanmean)(sc[q, k]) for k in range(K)}))
            EXP.append(np.asarray(X[np.flatnonzero(q)].mean(0)).ravel())
        if age == 'adult':                       # gene loadings over adult pseudobulks (layer-centred)
            df = pd.DataFrame({'t': sec, 'r': reg, 'l': lay})[ok & ~np.isin(reg, ['nan', 'None', 'remove', 'unassigned'])]
            gp = df.groupby(['t', 'r', 'l']).indices; base = df.index.values; keys = [k for k, v in gp.items() if len(v) >= 5]
            PBX = np.array([np.asarray(X[base[gp[k]]].mean(0)).ravel() for k in keys]); PBS = np.array([np.nanmean(sc[base[gp[k]]], 0) for k in keys]); kl = np.array([k[2] for k in keys])
            for l in LAYERS:
                q = kl == l; PBX[q] -= PBX[q].mean(0); PBS[q] -= PBS[q].mean(0)
            Xz = (PBX - PBX.mean(0)) / (PBX.std(0) + 1e-9); Sz = (PBS - PBS.mean(0)) / PBS.std(0); LOAD = Xz.T @ Sz / len(Sz)
        print(M.key, age, 'done', flush=True)
    S = pd.DataFrame(rows); S['ord'] = S.age.map(ORD); EXP = np.array(EXP)
    EZ = (EXP - EXP.mean(0)) / (EXP.std(0) + 1e-9); NOLOC = np.array([not g.startswith('LOC') for g in genes])
    G, R = [], []; dev = S.age != 'adult'
    for k in range(K):
        o_ = np.argsort(LOAD[:, k]); lo = [i for i in o_ if NOLOC[i]][:30]; hi = [i for i in o_[::-1] if NOLOC[i]][:30]
        S[f'hi{k + 1}'] = EZ[:, hi].mean(1); S[f'lo{k + 1}'] = EZ[:, lo].mean(1)
        for end, ix in (('positive', hi), ('negative', lo)):
            for rank, i in enumerate(ix):
                per_age = pd.Series(EZ[:, i]).groupby(S.age.values).mean()
                G.append(dict(axis=k + 1, reading=T.loc[k + 1].reading, end=end, rank=rank + 1, gene=genes[i], loading_r=LOAD[i, k],
                              rho_age_dev=spearmanr(S.ord[dev], EZ[dev.values, i])[0], **{f'z_{a}': per_age.get(a, np.nan) for a in AGES}))
        for nm, col in ((f'axis {k + 1}', f'axis{k + 1}'), (f'axis {k + 1} positive genes', f'hi{k + 1}'), (f'axis {k + 1} negative genes', f'lo{k + 1}')):
            R.append(dict(what=nm, reading=T.loc[k + 1].reading, rho_dev9=spearmanr(S.ord[dev], S[col][dev])[0], p_dev9=spearmanr(S.ord[dev], S[col][dev])[1],
                          rho_all36=spearmanr(S.ord, S[col])[0], **{f'median_{a}': S[S.age == a][col].median() for a in AGES}))
    S.to_csv(M.out(f'age_areal_samples{SUF}.csv'), index=False); pd.DataFrame(G).to_csv(M.out(f'age_areal_genes{SUF}.csv'), index=False)
    RR = pd.DataFrame(R); RR.to_csv(M.out(f'age_areal_rho{SUF}.csv'), index=False)
    return S, RR


def age_metrics(M):
    c = cfg(); Z = M.npz; kf = Z['keepf']; PG = Z['pb_G'][:K]; AG = AGES[:-1]
    ao = ad.read_h5ad(c.inputs(c.series['name']), backed='r').obs; Ea = np.load(c.emb(c.series['name'], M.adult_emb)).astype(np.float64)[:, kf]
    m_lay = {l: Ea[ao.layer.astype(str).values == l].mean(0) for l in LAYERS}; rows = []
    for i, age in enumerate(AG):
        o = ad.read_h5ad(c.inputs(f'dev{age}'), backed='r').obs; E = np.load(c.emb(f'dev{age}', M.dev_emb)).astype(np.float64)[:, kf]
        lay = o.layer.astype(str).values; sec = o.slice.astype(str).values; col = o.col.values.astype(int); sc = areal_scores(E, lay, m_lay, PG)
        for s in dict.fromkeys(sec):
            q = (sec == s) & np.isin(lay, LAYERS); r = dict(age=age, ord=i, sample=s)
            for k in range(K):
                v = sc[q, k]; r[f'mean_{k + 1}'] = v.mean(); r[f'median_{k + 1}'] = np.median(v); r[f'trim20_{k + 1}'] = trim_mean(v, 0.2)
                r[f'layer_median_{k + 1}'] = np.mean([np.median(sc[(sec == s) & (lay == l), k]) for l in LAYERS[1:] if ((sec == s) & (lay == l)).sum() > 20])
                qc = (sec == s) & (col >= 0) & np.isin(lay, LAYERS[1:]); cm = pd.Series(sc[qc, k]).groupby(col[qc]).mean(); r[f'column_median_{k + 1}'] = cm.median()
            rows.append(r)
    S = pd.DataFrame(rows); out = []
    for st in ['mean', 'median', 'trim20', 'layer_median', 'column_median']:
        for k in range(K):
            cc = f'{st}_{k + 1}'; rho = spearmanr(S.ord, S[cc])[0]
            loo = [spearmanr(S.ord.drop(i), S[cc].drop(i))[0] for i in S.index]
            out.append(dict(statistic=st, axis=k + 1, rho=rho, loo_min=min(loo), loo_max=max(loo), **{f'median_{a}': S[S.age == a][cc].median() for a in AG}))
    R = pd.DataFrame(out); R.to_csv(M.out('age_areal_metrics.csv'), index=False)
    return R
