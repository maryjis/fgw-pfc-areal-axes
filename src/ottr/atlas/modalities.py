"""E1 / E2 of the validation plan: our axes against the MRI, fMRI, connectivity and task maps carried onto the atlas segments by Huang
et al. 2026 (segments_meta.csv), and against the axes of those modalities.   python -m ottr.atlas.modalities
E1 regions  30 PFC regions (feature mean over the atlas segments); ours = mu of axes 1-5 and the gene scores (PrAl, axes 1-5). Spearman and
            Pearson, bootstrap CI (regions as independent: too narrow), leave-one-out, spatial nulls on OUR map (Moran 'pair' and variogram,
            3D centroids), partial Spearman given axis 1 (axes 2-5), and the feature from segments outside T402-T428 only (>= 5 segments).
            Primary family (fixed before the run): mu axis1-5 x {Myelinmap, FG1, FG2, SG1, SG2} = 25 tests, BH per null. Exploratory: the rest.
E1 segments atlas segments T408-T428, value of our axis on the segment (structure.segments), variogram surrogates (1000, k fitted and 200).
E2          PCA of the 30 regions x MRI features (Myelinmap, thickness of L2-L6: each feature z-scored); FG1/FG2, SG1/SG2 as given. Match of
            an axis with a modality = best |rho| over its components, p against the maximum over components under the surrogates of the axis.
Map signs of gradients and PCs are arbitrary. -> results/tables/validation/{e1_regions,e1_segments,e2_components,e2_match}.csv"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata
from statsmodels.stats.multitest import multipletests
from ..config import cfg, ensure
from ..registry import axes_model
from ..genescore import significance as SG
from . import structure as ST

PRIMARY = ['Myelinmap', 'FG1', 'FG2', 'SG1', 'SG2']; LAYER = [f'{l}-Layer-Thickness' for l in ST.LAYERS] + ['thickness_L2-L6', 'L4_fraction']
NET = ['VenSoma', 'DorSoma', 'ForPole', 'ParapHipp', 'OFC', 'AudIns', 'FPN', 'DMN', 'MedV1', 'LatV1', 'HighVisA', 'HighVisB', 'HighVisC', 'ACC', 'Premotor']
EXPLO = ['Hierachical', 'Animation-Task', 'Vocal-Task', 'Visual-Task'] + [f'{n}-Network' for n in NET] + LAYER; FEATS = PRIMARY + EXPLO; NS = 5000; NB = 5000
MODALITY = {'MRI': ['Myelinmap'] + [f'{l}-Layer-Thickness' for l in ST.LAYERS], 'FG': ['FG1', 'FG2'], 'SG': ['SG1', 'SG2']}


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); f = np.isfinite(p); q[f] = multipletests(p[f], method='fdr_bh')[1]; return q


def ours():
    c = cfg(); Z = axes_model('fgw2d_factors').npz; mu = pd.DataFrame(Z['pb_mu'][:, :5], index=Z['pb_regions'], columns=[f'mu axis{k}' for k in range(1, 6)])
    G = pd.read_csv(c.tables('gene_axes/final_scores_regions.csv'), index_col=0); s = ['PrAl'] + [f'axis{k}' for k in range(1, 6)]
    return mu.join(G[[f'{x}_adultref' for x in s]].rename(columns=lambda x: 'genes ' + x.replace('_adultref', '')))


def region_table(M):
    low = {r.lower(): r for r in M.region.dropna().unique()}; O = ours(); ren = {r: low.get(r.lower(), r) for r in O.index}
    A = M.groupby('region')[FEATS].mean(); Out = M[~M.in_series]; n = Out.groupby('region').size(); B = Out.groupby('region')[FEATS].mean().loc[n[n >= ST.MIN_OUT].index]
    f = lambda T: T.reindex([ren[r] for r in O.index]).set_axis(O.index)
    return O, f(A), f(B)


def e1_regions(M, rng):
    O, A, B = region_table(M); C = SG.marmoset_centroids(list(O.index)); out = []; sur = {}
    for o in O:
        x = O[o].values.astype(float); f = np.isfinite(x) & np.isfinite(C).all(1)
        for m in ('msr', 'variogram'): sur[o, m] = SG.rank_cols(SG.surrogates(x[f], C[f], NS, rng, m)[0]); 
        sur[o, 'f'] = f
    for o in O:
        x = O[o].values.astype(float); f = sur[o, 'f']
        for feat in FEATS:
            y = A[feat].values.astype(float); ok = f & np.isfinite(y); n = int(ok.sum())
            if not (ok == f).all(): continue
            r = SG.rho(x, y); bi = rng.integers(0, n, (NB, n)); xo, yo = x[ok], y[ok]; bs = np.array([np.corrcoef(rankdata(xo[i]), rankdata(yo[i]))[0, 1] for i in bi[:2000]]); bs = bs[np.isfinite(bs)]
            loo = [SG.rho(np.delete(xo, i), np.delete(yo, i)) for i in range(n)]; yo_out = B[feat].values.astype(float); go = f & np.isfinite(yo_out)
            d = dict(ours=o, feature=feat, family='primary' if (o.startswith('mu') and feat in PRIMARY) else 'exploratory', n=n, spearman=r, pearson=pearsonr(xo, yo)[0], ci_low=np.quantile(bs, .025), ci_high=np.quantile(bs, .975),
                     loo_min=min(loo), loo_max=max(loo), n_outside=int(go.sum()), rho_outside=SG.rho(x[go], yo_out[go]) if go.sum() >= 6 else np.nan)
            for m in ('msr', 'variogram'): nl = sur[o, m].T @ SG.unit(yo); d['p_' + m] = SG.p_two(nl, r)
            k = o.split('axis')[-1]
            if o.startswith('mu') and k != '1': d['partial_axis1'] = ST.partial_spearman(x[ok], yo, O['mu axis1'].values[ok][:, None].astype(float))
            out.append(d)
        print('regions', o, flush=True)
    D = pd.DataFrame(out)
    for m in ('msr', 'variogram'):
        D['q_' + m] = np.nan
        for fam, i in D.assign(g=D.family + '|' + D.ours.str[:2]).groupby('g').groups.items(): D.loc[i, 'q_' + m] = bh(D.loc[i, 'p_' + m])
    return D, O, A


def e1_segments(M):
    c = cfg(); V = pd.read_csv(c.tables('atlas/flatmap_fgw2d_factors/segment_values.csv'), index_col=0); O = ours(); D = M.loc[V.index]; reg = D.region
    xy = np.c_[D.AP.values.astype(float), 3.5 * D.Y_axis.values.astype(float)]; rows = []
    for k in range(1, 6):
        v = V[f'FGW 2D|areal|{k}'].values.astype(float); m = reg.map(O[f'mu axis{k}']).values.astype(float); f = np.isfinite(v) & np.isfinite(m); x = v * np.sign(np.corrcoef(v[f], m[f])[0, 1]); g = np.isfinite(x)
        S = ST.Surrogates(xy[g], ks=(1, 3, 6, 12, 25, 50, 100, 200)); kf = S.fit(x[g])
        for kk in sorted({kf, 200}):
            S.k = kk; dr = S.draw(x[g], 1000); rk = np.apply_along_axis(rankdata, 1, dr); rk = rk - rk.mean(1, keepdims=True); rk /= np.linalg.norm(rk, axis=1, keepdims=True)
            for feat in FEATS:
                y = D[feat].values.astype(float)[g]; w = np.isfinite(y)
                if w.sum() < 0.9 * len(y): continue
                y = np.where(w, y, np.nanmedian(y)); r = SG.rho(x[g], y); nl = rk @ SG.unit(y); rows.append(dict(axis=k, feature=feat, n=int(g.sum()), surrogate_k=kk, k_fitted=kk == kf, spearman=r, p=SG.p_two(nl, r)))
        print('segments axis', k, 'k', kf, flush=True)
    D = pd.DataFrame(rows); D['family'] = np.where(D.feature.isin(PRIMARY), 'primary', 'exploratory'); D['q'] = np.nan
    for _, i in D.groupby(['family', D.surrogate_k == 200]).groups.items(): D.loc[i, 'q'] = bh(D.loc[i, 'p'])
    return D


def e2(O, A, rng):
    comp = {}; info = []
    for mod, fs in MODALITY.items():
        X = A[fs].values.astype(float)
        if mod == 'MRI':
            Z = (X - X.mean(0)) / X.std(0); U, s, Vt = np.linalg.svd(Z, full_matrices=False); comp[mod] = pd.DataFrame(U[:, :3] * s[:3], index=A.index, columns=['MRI-PC1', 'MRI-PC2', 'MRI-PC3'])
            for j in range(3): info.append(dict(modality=mod, component=f'MRI-PC{j + 1}', share=(s[j] ** 2 / (s ** 2).sum()), **{f'load {f}': Vt[j, i] for i, f in enumerate(fs)}))
        else: comp[mod] = A[fs]
    C = SG.marmoset_centroids(list(O.index)); rows = []
    for k in range(1, 6):
        x = O[f'mu axis{k}'].values.astype(float)
        for m in ('msr', 'variogram'):
            Sx = SG.rank_cols(SG.surrogates(x, C, NS, rng, m)[0])
            for mod, T in comp.items():
                R = np.array([SG.rho(x, T[j].values) for j in T]); N = np.abs(np.array([Sx.T @ SG.unit(T[j].values) for j in T])); b = int(np.abs(R).argmax())
                rows.append(dict(axis=k, modality=mod, null=m, best=T.columns[b], rho=R[b], p_max=(1 + (N.max(0) >= abs(R[b]) - 1e-12).sum()) / (NS + 1), **{f'rho {j}': r for j, r in zip(T.columns, R)}))
    D = pd.DataFrame(rows); D['q'] = np.nan
    for _, i in D.groupby('null').groups.items(): D.loc[i, 'q'] = bh(D.loc[i, 'p_max'])
    return pd.DataFrame(info), D


def null_check(O, rng, n=100, length=0.4):
    """Calibration: independent smooth Gaussian maps (exponential covariance, length x median distance) against each other; share p < 0.05."""
    from scipy.spatial.distance import cdist
    C = SG.marmoset_centroids(list(O.index)); D = cdist(C, C); m = len(C); Lc = np.linalg.cholesky(np.exp(-D / (length * np.median(D[np.triu_indices(m, 1)]))) + 1e-8 * np.eye(m)); p = {'msr': [], 'variogram': []}
    for _ in range(n):
        x, y = Lc @ rng.standard_normal(m), Lc @ rng.standard_normal(m)
        for k in p: p[k].append(SG.p_two(SG.rank_cols(SG.surrogates(x, C, 300, rng, k)[0]).T @ SG.unit(y), SG.rho(x, y)))
    return {k: float(np.mean(np.array(v) < .05)) for k, v in p.items()}


def run():
    c = cfg(); out = ensure(c.tables('validation')); M = ST.meta(); rng = np.random.default_rng(31)
    D, O, A = e1_regions(M, rng); D.to_csv(os.path.join(out, 'e1_regions.csv'), index=False); print('null check (share p<0.05):', null_check(O, rng), flush=True)
    old = pd.read_csv(c.tables('atlas/axes_vs_structure_regions.csv')); old = old[old.source == 'all atlas segments'].set_index(['ours', 'feature']).spearman
    j = D[D.feature.isin(ST.FEATURES)].set_index(['ours', 'feature']).spearman; com = j.index.intersection(old.index); print('max diff to axes_vs_structure_regions:', float((j[com] - old[com]).abs().max()), 'n', len(com), flush=True)
    I, E = e2(O, A, rng); I.to_csv(os.path.join(out, 'e2_components.csv'), index=False); E.to_csv(os.path.join(out, 'e2_match.csv'), index=False); print('e2 done', flush=True)
    e1_segments(M).to_csv(os.path.join(out, 'e1_segments.csv'), index=False); print('done', flush=True)


if __name__ == '__main__': run()
