"""How significant are the correlations we report? Null models and robustness for
(a) the cross-species comparison (ottr.genescore.species: macaque / human frontal regions vs our marmoset values),
(b) the region-level comparison of our axes with the atlas structure (ottr.atlas.structure: myelin, layer fractions),
(c) the segment-level spatial p of ottr.atlas.structure, re-done with longer autocorrelation scales (axis 4),
(d) snRNA vs Stereo-seq over 7 adult regions (exact permutation).
python -m ottr.genescore.significance [--nsets 10000] [--nsurr 10000]

1. Gene null (cross-species). For each score (PrAl, axes 1-5) and species: random gene sets with, per end, the same number of
   genes as were found in the species, each real gene replaced by a random gene of the same species expression decile
   (mean log1p CP10k over all regions for macaque, mean value for human), drawn from a pool:
     model  = the 1937 genes of the model (the encoder sets were picked from them),
     genome = all marmoset genes measured in the species.
   The random set is scored with the same formula as species.scores (z over all regions of the species, z(+) - z(-)) and
   correlated (Spearman) with the same marmoset target: frontal regions vs our mu, and all regions vs the species' Pr-Al
   reference. p = (1 + #|rho_null| >= |rho|) / (B + 1) (two-sided: the sign of a random end is arbitrary).
2. Spatial null (marmoset geometry). Region centroids from the atlas segments (Segment_genes_marmoset meta; AP is the section
   index, ~0.27 mm, DV / ML ~0.027 mm per unit -> 3D (10 AP, DV, ML); sensitivity: the authors' flatmap (AP, 3.5 Y)).
   Surrogates of OUR map (mu of an axis, or our Stereo-seq gene score) over the 30 regions, preserving its spatial
   autocorrelation:
     msr        Moran spectral randomisation (Wagner & Dray 2015): random signs of the coefficients on the eigenvectors of the
                doubly centred inverse-distance weights (Moran's I preserved exactly);
     variogram  as BrainSMASH (Burt et al. 2020): permute, smooth with an exponential kernel over the k nearest regions,
                rescale to match the empirical variogram (k chosen by the fit), values rank-matched to the original.
   For the species, a surrogate is carried through data/species_region_map.tsv exactly as mu (mean over the mapped marmoset
   regions) and correlated with the fixed species score. For the structure, it is correlated with the atlas feature.
3. Robustness of every frontal / structure correlation: bootstrap over rows (95% percentile CI), leave-one-out and
   leave-two-out (lowest rho), leaving out all rows mapped to one marmoset region, duplicates collapsed (one point per set of
   marmoset target regions), exact name matches only. BH-FDR over all tests of one kind.
-> results/tables/gene_axes/significance/{gene_null,spatial_null,robustness,segments_scales,snrna_exact}.csv,
   results/pictures/significance_species.png"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import cdist
from ..config import cfg, ensure, ROOT
from ..registry import axes_model
from . import species as SPC

SCORES = SPC.SCORES; SPECIES = SPC.SPECIES; AX = lambda s: 1 if s == 'PrAl' else int(s[-1])


def rho(x, y):
    f = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(rankdata(x[f]), rankdata(y[f]))[0, 1])


def rank_cols(A):
    """Column-wise ranks (average ties) of a 2D array, centred and scaled to unit norm (Spearman = dot product)."""
    R = np.apply_along_axis(rankdata, 0, A); R = R - R.mean(0); return R / np.linalg.norm(R, axis=0)


def unit(v):
    r = rankdata(v); r = r - r.mean(); return r / np.linalg.norm(r)


def p_two(null, obs):
    return (1 + int((np.abs(null) >= abs(obs) - 1e-12).sum())) / (len(null) + 1)


# ---------------------------------------------------------------- 1. gene null
def species_matrix(sp_):
    """regions x genes matrix in the scale the score averages (log1p CP10k for counts, values as they are otherwise)."""
    X, genes, R, kind = SPC.load(sp_); X = X.toarray().astype(np.float64)
    if kind == 'int32': X = np.log1p(X * (1e4 / X.sum(1, keepdims=True)))
    _, first = np.unique(genes, return_index=True); first = np.sort(first)              # duplicated symbols: keep the first
    return X[:, first], genes[first], R


def set_scores(Y, idx):
    """Y regions x genes, idx sets x genes (index arrays, one end): z over regions of the mean over the genes of each set."""
    m = Y[:, idx].mean(2); return (m - m.mean(0)) / m.std(0)


def gene_null(nsets, seed=0):
    c = cfg(); rng = np.random.default_rng(seed); S = SPC.gene_sets(); out = []
    pool_model = set(pd.read_csv(os.path.join(c.work, 'genescore', 'encoder_axes.csv'), index_col=0).index.str.upper())
    marm = set(g.upper() for g in open(os.path.join(c.work, 'species', 'marmoset', 'genes.txt')).read().split('\n') if g)
    FR = pd.read_csv(c.tables('gene_axes/species/compare_frontal.csv'), dtype={'their_region': str}); nulls = {}
    for sp_ in SPECIES:
        Y, genes, R = species_matrix(sp_); pos = {g: i for i, g in enumerate(genes)}; mexp = Y.mean(0)
        expressed = mexp > 0; regions = R.region.astype(str).values
        F = FR[FR.species == sp_]; rows = np.array([np.where(regions == r)[0][0] for r in F.their_region])
        ref = R[SPC.REF[sp_]].values.astype(float); fr = np.isfinite(ref)
        for s in SCORES:
            ends = {}
            for side in ('Pr', 'Al'):
                ends[side] = np.array([pos[g] for g in S[s][side] if g in pos])
            obs_score = set_scores(Y, ends['Pr'][None])[:, 0] - set_scores(Y, ends['Al'][None])[:, 0]
            k = AX(s); tgt = F[f'mu_axis{k}'].values.astype(float); ft = np.isfinite(tgt); rw, tgt = rows[ft], tgt[ft]   # Iai -> OPro has no mu
            obs = dict(frontal=rho(obs_score[rw], tgt), reference=rho(obs_score[fr], ref[fr]))
            for pool_name, pool in (('model', pool_model), ('genome', marm)):
                cand = np.array([i for g, i in pos.items() if g in pool and expressed[i]])
                bins = np.digitize(np.log10(mexp[cand]), np.quantile(np.log10(mexp[cand]), np.linspace(0, 1, 11)[1:-1]))
                by_bin = {b: cand[bins == b] for b in range(10)}
                gbin = lambda gi: np.digitize(np.log10(mexp[gi]), np.quantile(np.log10(mexp[cand]), np.linspace(0, 1, 11)[1:-1]))
                ut, ur = unit(tgt), unit(ref[fr]); nf, nr = [], []
                for start in range(0, nsets, 250):
                    b = min(250, nsets - start); Z = {}
                    for side in ('Pr', 'Al'):
                        gb = gbin(ends[side]); idx = np.empty((b, len(gb)), int)
                        for j, bb in enumerate(gb): idx[:, j] = rng.choice(by_bin[bb], b)
                        Z[side] = set_scores(Y, idx)
                    sc = Z['Pr'] - Z['Al']
                    nf.append(rank_cols(sc[rw]).T @ ut); nr.append(rank_cols(sc[fr]).T @ ur)
                nf, nr = np.concatenate(nf), np.concatenate(nr); nulls[(sp_, s, pool_name)] = nf
                for comp, null, o in (('frontal vs our mu', nf, obs['frontal']), (f'all regions vs {SPC.REF[sp_]}', nr, obs['reference'])):
                    out.append(dict(species=sp_, score=s, comparison=comp, pool=pool_name, n_pool=len(cand),
                                    n_genes_pos=len(ends['Pr']), n_genes_neg=len(ends['Al']), rho=o,
                                    null_mean=null.mean(), null_abs_q95=np.quantile(np.abs(null), 0.95),
                                    p_two=p_two(null, o), p_one=(1 + int((np.sign(o) * null >= abs(o) - 1e-12).sum())) / (len(null) + 1),
                                    rank_pct=100 * (np.abs(null) < abs(o)).mean()))
            print('gene null', sp_, s, 'done', flush=True)
    return pd.DataFrame(out), nulls


# ---------------------------------------------------------------- 2. spatial null
def marmoset_centroids(regions, geometry='3d'):
    from ..atlas.structure import meta
    M = meta(); g = M.groupby('region')
    C = g[['AP', 'DV', 'ML']].mean() if geometry == '3d' else g[['AP', 'Y_axis']].mean()
    C = C * ([10, 1, 1] if geometry == '3d' else [1, 3.5]); low = {r.lower(): r for r in C.index}          # atlas writes Opro for OPro
    return C.reindex([low.get(r.lower(), r) for r in regions]).values


def msr(x, D, n, rng):
    """Moran spectral randomisation, 'pair' procedure (Wagner & Dray 2015): eigenvectors of the doubly centred inverse-distance
    weights sorted by eigenvalue; the coefficients of x on each consecutive pair are rotated by a random angle (plus random
    signs), so Moran's I is kept up to the eigenvalue gap within a pair and dominant patterns get mixed with their neighbours
    (with signs alone a map lying on one eigenvector only flips sign). Values rank-matched to the original."""
    W = np.where(D > 0, 1 / np.where(D > 0, D, 1), 0); m = len(x); H = np.eye(m) - 1 / m
    lam, V = np.linalg.eigh(H @ W @ H); keep = np.abs(V.sum(0)) < 1e-6 * np.sqrt(m)        # drop the constant direction
    lam, V = lam[keep], V[:, keep]; o = np.argsort(-lam); V = V[:, o]; cx = V.T @ (x - x.mean()); q = len(cx)
    C = np.repeat(cx[:, None], n, 1) * rng.choice([-1, 1], (q, n))
    for i in range(0, q - 1, 2):
        t = rng.uniform(0, 2 * np.pi, n); a, b = C[i].copy(), C[i + 1].copy()
        C[i], C[i + 1] = np.cos(t) * a - np.sin(t) * b, np.sin(t) * a + np.cos(t) * b
    S = V @ C; srt = np.sort(x); out = np.empty_like(S)
    for j in range(n): out[np.argsort(S[:, j]), j] = srt
    return out


def variogram_surrogates(x, D, n, rng, pv=70, nh=10, ks=None):
    """BrainSMASH-style (Burt et al. 2020, Base): permute, smooth with an exponential kernel over the k nearest regions, rescale
    so the variogram matches the empirical one (linear fit over distance bins up to the pv-th percentile), rank-match values.
    pv is high because there are only ~30 regions (BrainSMASH's default 25 is for hundreds of parcels)."""
    m = len(x); iu = np.triu_indices(m, 1); d = D[iu]; keep = d <= np.percentile(d, pv)
    edges = np.linspace(d[keep].min(), d[keep].max(), nh + 1); b = np.clip(np.digitize(d[keep], edges) - 1, 0, nh - 1)
    vg = lambda v: np.array([np.mean(0.5 * (v[iu[0]][keep] - v[iu[1]][keep])[b == h] ** 2) if (b == h).any() else np.nan for h in range(nh)])
    target = vg(x); order = np.argsort(D, 1); ks = ks or [k for k in (3, 5, 8, 12, 16, 20) if k < m]
    def smooth(p, k):
        nn = order[:, :k]; dd = np.take_along_axis(D, nn, 1); w = np.exp(-dd / (dd.max(1, keepdims=True) + 1e-12))
        return (w * p[nn]).sum(1) / w.sum(1)
    def one(k):
        s = smooth(rng.permutation(x), k); v = vg(s); f = np.isfinite(v) & np.isfinite(target)
        beta, alpha = np.polyfit(v[f], target[f], 1); out = np.sqrt(abs(beta)) * (s - s.mean()) + np.sqrt(abs(alpha)) * rng.standard_normal(m)
        return out, np.nansum((vg(out) - target) ** 2)
    err = {k: np.mean([one(k)[1] for _ in range(20)]) for k in ks}; kbest = min(err, key=err.get)
    srt = np.sort(x); S = np.empty((m, n))
    for i in range(n): s, _ = one(kbest); S[np.argsort(s), i] = srt
    return S, kbest


def surrogates(x, C, n, rng, method):
    D = cdist(C, C)
    if method == 'msr': return msr(x, D, n, rng), None
    return variogram_surrogates(x, D, n, rng)


def spatial_null(nsurr, seed=1):
    c = cfg(); rng = np.random.default_rng(seed); Z = axes_model('fgw2d_factors').npz
    mu = pd.DataFrame(Z['pb_mu'][:, :5], index=Z['pb_regions'], columns=[f'mu_axis{k}' for k in range(1, 6)])
    st = pd.read_csv(c.tables('gene_axes/final_scores_regions.csv'), index_col=0)
    ours = mu.join(st[[f'{s}_adultref' for s in SCORES]].rename(columns=lambda s: 'stereo_' + s.replace('_adultref', '')), how='outer')
    regs = list(ours.index); FR = pd.read_csv(c.tables('gene_axes/species/compare_frontal.csv'), dtype={'their_region': str})
    from ..atlas.structure import meta, FEATURES, SERIES, MIN_OUT
    M = meta(); A = M.groupby('region')[FEATURES].mean(); O = M[~M.in_series]; no = O.groupby('region').size()
    B = O.groupby('region')[FEATURES].mean().loc[no[no >= MIN_OUT].index]; out = []
    for geometry in ('3d', 'flatmap'):
        C = marmoset_centroids(regs, geometry)
        for col in ours.columns:
            x = ours[col].values.astype(float); f = np.isfinite(x) & np.isfinite(C).all(1)
            for method in ('msr', 'variogram'):
                Sx = np.full((len(regs), nsurr), np.nan); S, kb = surrogates(x[f], C[f], nsurr, rng, method); Sx[f] = S
                Sdf = pd.DataFrame(Sx, index=regs)
                # species: carry each surrogate through the region map exactly as mu
                for sp_ in SPECIES:
                    F = FR[FR.species == sp_]; s_ = 'PrAl' if col == 'mu_axis1' else col.split('_', 1)[1]
                    targets = [s_] if s_ in SCORES else []
                    if col == 'mu_axis1': targets = ['PrAl', 'axis1']
                    for s in targets:
                        y = F[f'species_{s}'].values.astype(float); xo = F[col].values.astype(float)
                        Smap = np.vstack([Sdf.reindex(o.split(',')).mean(0).values for o in F.our_regions])
                        ok = np.isfinite(xo) & np.isfinite(y) & np.isfinite(Smap).all(1); null = rank_cols(Smap[ok]).T @ unit(y[ok]); o = rho(xo, y)
                        out.append(dict(test='species frontal', species=sp_, ours=col, other=f'{sp_} {s}', n=int(ok.sum()), geometry=geometry,
                                        method=method, k=kb, rho=o, p_spatial=p_two(null, o), null_abs_q95=np.quantile(np.abs(null), 0.95)))
                # structure: atlas feature means per region
                if col.startswith('mu'):
                    for src, Fe in (('all atlas segments', A), (f'outside T{SERIES[0]}-T{SERIES[1]}', B)):
                        for feat in FEATURES:
                            y = Fe[feat].reindex(regs).values.astype(float); ok = f & np.isfinite(y)
                            if ok.sum() < 6: continue
                            null = rank_cols(Sx[ok]).T @ unit(y[ok]); o = rho(x, y)
                            out.append(dict(test=f'structure ({src})', species='marmoset', ours=col, other=feat, n=int(ok.sum()), geometry=geometry,
                                            method=method, k=kb, rho=o, p_spatial=p_two(null, o), null_abs_q95=np.quantile(np.abs(null), 0.95)))
            print('spatial', geometry, col, 'done', flush=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 3. robustness
def robust(x, y, labels, groups, exact, nboot, rng):
    f = np.isfinite(x) & np.isfinite(y); x, y = x[f], y[f]; labels, groups, exact = labels[f], groups[f], exact[f]; n = len(x); o = rho(x, y)
    bi = rng.integers(0, n, (nboot, n)); bs = np.array([rho(x[i], y[i]) for i in bi]); bs = bs[np.isfinite(bs)]
    loo = np.array([rho(np.delete(x, i), np.delete(y, i)) for i in range(n)])
    l2o = min((rho(np.delete(x, [i, j]), np.delete(y, [i, j])), (i, j)) for i, j in itertools.combinations(range(n), 2))
    G = pd.DataFrame({'x': x, 'y': y, 'g': groups}).groupby('g').mean(); lgo = {}
    for gg in set(itertools.chain.from_iterable(g.split(',') for g in groups)):
        keep = np.array([gg not in g.split(',') for g in groups])
        if keep.sum() >= 5: lgo[gg] = rho(x[keep], y[keep])
    worst_g = min(lgo, key=lgo.get) if lgo else None
    return dict(n=n, rho=o, ci_low=np.quantile(bs, 0.025), ci_high=np.quantile(bs, 0.975), loo_min=loo.min(), loo_min_without=labels[loo.argmin()],
                loo_max=loo.max(), l2o_min=l2o[0], l2o_min_without=f'{labels[l2o[1][0]]}+{labels[l2o[1][1]]}',
                lgo_min=lgo.get(worst_g, np.nan), lgo_min_without=worst_g, n_collapsed=len(G), rho_collapsed=rho(G.x.values, G.y.values),
                n_exact=int(exact.sum()), rho_exact=rho(x[exact], y[exact]) if exact.sum() >= 5 else np.nan)


def robustness(nboot, seed=2):
    c = cfg(); rng = np.random.default_rng(seed); FR = pd.read_csv(c.tables('gene_axes/species/compare_frontal.csv'), dtype={'their_region': str}); out = []
    for sp_ in SPECIES:
        F = FR[FR.species == sp_]
        for s in SCORES:
            for ours in (f'mu_axis{AX(s)}', f'stereo_{s}'):
                r = robust(F[ours].values.astype(float), F[f'species_{s}'].values.astype(float), F.their_region.values, F.our_regions.values,
                           (F.match == 'exact').values, nboot, rng)
                out.append(dict(test='species frontal', species=sp_, ours=ours, other=f'{sp_} {s}', **r))
    from ..atlas.structure import meta, FEATURES
    M = meta(); A = M.groupby('region')[FEATURES].mean(); Z = axes_model('fgw2d_factors').npz
    mu = pd.DataFrame(Z['pb_mu'][:, :5], index=Z['pb_regions']); regs = np.array(mu.index)
    for k in range(1, 6):
        for feat in FEATURES:
            y = A[feat].reindex(regs).values.astype(float)
            r = robust(mu[k - 1].values.astype(float), y, regs, regs, np.zeros(len(regs), bool), nboot, rng)
            out.append(dict(test='structure (all atlas segments)', species='marmoset', ours=f'mu_axis{k}', other=feat, **r))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 4. segments: axis 4 and longer scales
def segments_scales(nsurr, seed=3):
    from ..atlas import structure as ST
    c = cfg(); M = ST.meta(); V = pd.read_csv(c.tables('atlas/flatmap_fgw2d_factors/segment_values.csv'), index_col=0)
    Z = axes_model('fgw2d_factors').npz; mu = pd.DataFrame(Z['pb_mu'][:, :5], index=Z['pb_regions'], columns=ST.AXES)
    D = M.loc[V.index]; xy = np.c_[D.AP.values.astype(float), 3.5 * D.Y_axis.values.astype(float)]; out = []
    for k in ST.AXES:
        v = V[f'FGW 2D|areal|{k}'].values.astype(float); m = D.region.map(mu[k]).values; f = np.isfinite(v) & np.isfinite(m)
        x = v * np.sign(np.corrcoef(v[f], m[f])[0, 1]); g = np.isfinite(x)
        S = ST.Surrogates(xy[g], ks=(6, 25, 50, 100, 200), seed=seed); kfit = S.fit(x[g])
        for kk in sorted({kfit, 25, 50, 100, 200}):
            S.k = kk; sur = S.draw(x[g], nsurr)
            for feat in ST.FEATURES:
                y = D[feat].values.astype(float)[g]; h = np.isfinite(y); o = rho(x[g][h], y[h])
                null = rank_cols(sur[:, h].T).T @ unit(y[h])
                out.append(dict(axis=k, feature=feat, k=kk, k_is_fit=kk == kfit, rho=o, p_spatial=p_two(null, o), null_abs_q95=np.quantile(np.abs(null), 0.95)))
        print('segments axis', k, 'done (fit k =', kfit, ')', flush=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 5. snRNA, 7 regions: exact permutation
def snrna_exact():
    c = cfg(); T = pd.read_csv(c.tables('gene_axes/snrna/compare_stereo_adult.csv')); regs = [s[3:] for s in T.columns if s.startswith('sn_')]
    perms = np.array(list(itertools.permutations(range(len(regs))))); out = []
    for _, r in T.iterrows():
        x = np.array([r[f'sn_{g}'] for g in regs], float); y = np.array([r[f'st_{g}'] for g in regs], float); o = rho(x, y)
        rx = rankdata(x); null = np.array([np.corrcoef(rx[p], rankdata(y))[0, 1] for p in perms])
        # A24A32 and A32 overlap: also without A32
        k = [i for i, g in enumerate(regs) if g != 'A32']; o6 = rho(x[k], y[k])
        p6 = np.array(list(itertools.permutations(range(len(k))))); n6 = np.array([np.corrcoef(rankdata(x[k])[p], rankdata(y[k]))[0, 1] for p in p6])
        out.append(dict(group=r.group, score=r.score, n=len(regs), rho=o, p_exact=float((np.abs(null) >= abs(o) - 1e-12).mean()),
                        rho_without_A32=o6, p_exact_without_A32=float((np.abs(n6) >= abs(o6) - 1e-12).mean())))
    return pd.DataFrame(out)


def fdr(df, p, by):
    from statsmodels.stats.multitest import multipletests
    df[p.replace('p', 'q', 1)] = np.nan
    for _, i in df.groupby(by).groups.items(): df.loc[i, p.replace('p', 'q', 1)] = multipletests(df.loc[i, p], method='fdr_bh')[1]
    return df


def figure(G, Sp, Rb, nulls):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    c = cfg(); INK, INK2, GRID = '#14161a', '#5b5e66', '#e4e3df'; CS = {'macaque': '#2a78d6', 'human': '#eb6834'}
    fig = plt.figure(figsize=(17, 9.5), facecolor='white'); gs = fig.add_gridspec(2, 4, width_ratios=[1.9, 1, 1, 1], hspace=0.5, wspace=0.35)
    # forest: frontal species score vs our mu, rho with bootstrap CI; p of the gene null (model pool) and the spatial null
    a = fig.add_subplot(gs[:, 0]); y = 0; ticks, lab = [], []
    for s in SCORES[::-1]:
        for sp_ in SPECIES[::-1]:
            r = Rb[(Rb.species == sp_) & (Rb.ours == f'mu_axis{AX(s)}') & (Rb.other == f'{sp_} {s}')].iloc[0]
            g = G[(G.species == sp_) & (G.score == s) & (G.pool == 'model') & (G.comparison == 'frontal vs our mu')].iloc[0]
            q = Sp[(Sp.species == sp_) & (Sp.ours == f'mu_axis{AX(s)}') & (Sp.other == f'{sp_} {s}') & (Sp.geometry == '3d') & (Sp.method == 'msr')].iloc[0]
            a.plot([r.ci_low, r.ci_high], [y, y], color=CS[sp_], lw=2.2, solid_capstyle='round'); a.plot(r.rho, y, 'o', color=CS[sp_], ms=7, mec='white')
            a.text(1.08, y, f'{g.p_two:.4f}', va='center', fontsize=8, color=INK2); a.text(1.55, y, f'{q.p_spatial:.4f}', va='center', fontsize=8, color=INK2)
            ticks.append(y); lab.append(f'{"PrAl" if s == "PrAl" else "Axis " + s[-1]} · {sp_}'); y += 1
        y += 0.6
    a.axvline(0, color=GRID, lw=1); a.set_yticks(ticks); a.set_yticklabels(lab, fontsize=9, color=INK); a.set_xlim(-1, 2.0); a.set_xticks([-1, -0.5, 0, 0.5, 1])
    a.text(1.04, y - 0.2, 'genes p', fontsize=8, fontweight='bold', color=INK); a.text(1.55, y - 0.2, 'spatial p', fontsize=8, fontweight='bold', color=INK)
    a.set_xlabel('Spearman ρ, frontal regions vs our μ (95% bootstrap CI)', fontsize=9, color=INK2)
    a.set_title('Cross-species: ρ, CI and two nulls', loc='left', fontsize=11, fontweight='bold', color=INK)
    for s_ in ('top', 'right', 'left'): a.spines[s_].set_visible(False)
    a.tick_params(length=0, labelsize=8.5)
    # gene-null histograms
    for j, (sp_, s) in enumerate([(sp, sc) for sc in ('axis1', 'axis3') for sp in SPECIES]):
        ax = fig.add_subplot(gs[j // 2, 1 + j % 2])
        for pool, colr in (('genome', '#c9cbd0'), ('model', CS[sp_])):
            ax.hist(nulls[(sp_, s, pool)], bins=60, range=(-1, 1), color=colr, alpha=0.75 if pool == 'model' else 0.9, label=f'random sets, {pool} pool')
        o = G[(G.species == sp_) & (G.score == s) & (G.pool == 'model') & (G.comparison == 'frontal vs our mu')].iloc[0]
        ax.axvline(o.rho, color=INK, lw=2); ax.set_title(f'{sp_.capitalize()} · Axis {s[-1]} genes\nρ = {o.rho:+.2f}, gene-null p = {o.p_two:.4f}', loc='left', fontsize=10, fontweight='bold', color=INK)
        ax.set_xlabel('ρ with our μ (frontal)', fontsize=8.5, color=INK2); ax.legend(frameon=False, fontsize=7.5, loc='upper left')
        for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
        ax.tick_params(length=0, labelsize=8, colors=INK2)
    # spatial null summary for structure (mu vs L5 / myelin)
    ax = fig.add_subplot(gs[:, 3]); T = Sp[(Sp.test == 'structure (all atlas segments)') & (Sp.geometry == '3d')]
    pairs = [('mu_axis1', 'Myelinmap'), ('mu_axis1', 'L3-Layer-Thickness'), ('mu_axis3', 'L5-Layer-Thickness'), ('mu_axis3', 'L4-Layer-Thickness'), ('mu_axis4', 'L6-Layer-Thickness')]
    for i, (m_, f_) in enumerate(pairs):
        for d, (meth, colr) in enumerate((('msr', '#2a78d6'), ('variogram', '#eb6834'))):
            r = T[(T.ours == m_) & (T.other == f_) & (T.method == meth)].iloc[0]
            ax.barh(i + 0.38 * d, -np.log10(r.p_spatial), height=0.36, color=colr, label=meth if i == 0 else None)
            ax.text(-np.log10(r.p_spatial) + 0.05, i + 0.38 * d, f'ρ {r.rho:+.2f}, p {r.p_spatial:.3f}', va='center', fontsize=7.5, color=INK2)
    ax.axvline(-np.log10(0.05), color=INK2, ls=(0, (4, 3)), lw=1); ax.set_yticks(np.arange(len(pairs)) + 0.19)
    ax.set_yticklabels([f'{m_.replace("mu_", "").replace("axis", "Axis ")} × {f_.replace("-Layer-Thickness", " fraction").replace("Myelinmap", "myelin")}' for m_, f_ in pairs], fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlabel('−log10 p (spatial null, 30 regions)', fontsize=9, color=INK2); ax.legend(frameon=False, fontsize=8)
    ax.set_title('Marmoset regions vs atlas structure', loc='left', fontsize=11, fontweight='bold', color=INK)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.tick_params(length=0)
    fig.suptitle('How significant are the correlations? Gene-set null (random genes, matched expression) and spatial null (marmoset region geometry)',
                 x=0.01, ha='left', fontsize=12, fontweight='bold', color=INK)
    p = c.pictures('significance_species.png'); fig.savefig(p, dpi=150, facecolor='white', bbox_inches='tight'); plt.close(fig); print('saved', p)


def run(nsets=10000, nsurr=10000):
    c = cfg(); O = ensure(c.tables('gene_axes/significance')); pd.set_option('display.width', 250)
    Sn = snrna_exact(); Sn.to_csv(os.path.join(O, 'snrna_exact.csv'), index=False); print(Sn.round(3).to_string(index=False), flush=True)
    Rb = robustness(nboot=10000); Rb.to_csv(os.path.join(O, 'robustness.csv'), index=False); print(Rb.round(3).to_string(index=False), flush=True)
    Sp = spatial_null(nsurr); Sp = fdr(Sp, 'p_spatial', ['test', 'geometry', 'method']); Sp.to_csv(os.path.join(O, 'spatial_null.csv'), index=False)
    print(Sp[Sp.geometry == '3d'].round(3).to_string(index=False), flush=True)
    G, nulls = gene_null(nsets); G = fdr(G, 'p_two', ['comparison', 'pool']); G.to_csv(os.path.join(O, 'gene_null.csv'), index=False)
    np.savez_compressed(os.path.join(c.work, 'genescore', 'gene_null_frontal.npz'), **{f'{a}|{b}|{d}': v for (a, b, d), v in nulls.items()})
    print(G.round(4).to_string(index=False), flush=True)
    Sg = segments_scales(1000); Sg.to_csv(os.path.join(O, 'segments_scales.csv'), index=False); print(Sg.round(3).to_string(index=False), flush=True)
    figure(G, Sp, Rb, nulls)




def eigen_maps():
    """Figure: the first three Moran eigenvectors of the 30 marmoset regions (3D geometry as spatial_null) next to mu of axes 1, 3, 4,
    drawn on the atlas segments (Segment_genes_marmoset meta): the authors' flatmap (section x position along the cortex) and an
    orthographic 3D view (top: AP x ML, front: ML x DV, side: AP x DV). Other cortex of the frontal sections in grey.
    -> results/pictures/spatial_eigenvectors_brain.png, tables/gene_axes/significance/spatial_eigenvectors.csv"""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from ..atlas.structure import meta
    c = cfg(); Z = axes_model('fgw2d_factors').npz; regs = list(Z['pb_regions']); C = marmoset_centroids(regs)
    D = cdist(C, C); W = np.where(D > 0, 1 / np.where(D > 0, D, 1), 0); m = len(regs); H = np.eye(m) - 1 / m
    lam, V = np.linalg.eigh(H @ W @ H); keep = np.abs(V.sum(0)) < 1e-6 * np.sqrt(m); lam, V = lam[keep], V[:, keep]; o = np.argsort(-lam); V = V[:, o]
    mu = Z['pb_mu'][:, :5]; T = pd.DataFrame({f'eig{i + 1}': V[:, i] for i in range(5)}, index=regs)
    for k in range(5): T[f'mu_axis{k + 1}'] = mu[:, k]
    for i in range(3):                                                     # sign of each eigenvector: as its matching axis
        j = [0, 2, 3][i]; T[f'eig{i + 1}'] *= np.sign(np.corrcoef(T[f'eig{i + 1}'], mu[:, j])[0, 1])
    share = {}
    for k in range(5):
        cx = V.T @ (mu[:, k] - mu[:, k].mean()); share[k + 1] = cx ** 2 / (cx ** 2).sum()
    T.to_csv(c.tables('gene_axes/significance/spatial_eigenvectors.csv'))
    M = meta(); low = {r.lower(): r for r in regs}; M['reg'] = M.region.map(lambda r: low.get(str(r).lower()))
    F = M[M.reg.notna()].copy(); ap_max = F.AP.max() + 2; B = M[M.reg.isna() & (M.AP <= ap_max)]
    INK, INK2, BG = '#14161a', '#5b5e66', '#e6e5e1'
    views = {'top': ('AP', 'ML', 'DV', 'AP (section) →  caudal', 'ML →  medial', 'top view'),
             'front': ('ML', 'DV', 'AP', 'ML →  medial', 'DV →  dorsal', 'front view (from rostral)'),
             'side': ('AP', 'DV', 'ML', 'AP (section) →  caudal', 'DV →  dorsal', 'side view (from lateral)')}
    rows = [('eig1', 'mu_axis1', 1, 'top', '1st eigenvector: lateral ↔ medial', 'Axis 1 (Pr–Al)'),
            ('eig2', 'mu_axis3', 3, 'front', '2nd eigenvector: dorsal ↔ ventral', 'Axis 3 (A24b ↔ OPAl)'),
            ('eig3', 'mu_axis4', 4, 'side', '3rd eigenvector: rostral ↔ caudal', 'Axis 4 (section effect)')]
    fig, ax = plt.subplots(3, 4, figsize=(21, 14.5), facecolor='white', gridspec_kw=dict(width_ratios=[1, 1, 1.15, 1.15]))
    cent = F.groupby('reg')[['AP', 'Y_axis']].mean()
    for r, (e, a_, k, vw, te, ta) in enumerate(rows):
        for cidx, (col, ttl) in enumerate([(e, te), (a_, ta), (e, te), (a_, ta)]):
            A = ax[r, cidx]; v = F.reg.map(T[col]).values; lim = np.nanmax(np.abs(T[col])); norm = TwoSlopeNorm(0, -lim, lim)
            if cidx < 2:                                                   # flatmap
                A.scatter(B.AP, B.Y_axis, s=5, c=BG, linewidths=0, rasterized=True)
                sc = A.scatter(F.AP, F.Y_axis, s=7, c=v, cmap='RdBu_r', norm=norm, linewidths=0, rasterized=True)
                for reg_, (x, y) in cent.iterrows():
                    A.annotate(reg_, (x, y), ha='center', va='center', fontsize=6.3, color=INK, alpha=0.85)
                A.set_xlabel('section (AP) →  caudal', fontsize=8.5, color=INK2); A.set_ylabel('position along the cortex (flatmap Y)', fontsize=8.5, color=INK2)
                A.set_xlim(0, ap_max)
            else:
                x_, y_, dep, xl, yl, vt = views[vw]; Fs = F.sort_values(dep, ascending=(vw != 'top'))   # painter's order: nearer on top
                if vw == 'side': Fs = F.sort_values('ML', ascending=False)     # lateral surface (low ML) drawn last
                if vw == 'front': Fs = F.sort_values('AP', ascending=False)    # rostral drawn last
                if vw == 'top': Fs = F.sort_values('DV', ascending=True)       # dorsal drawn last
                vv = Fs.reg.map(T[col]).values
                sc = A.scatter(Fs[x_], Fs[y_], s=9, c=vv, cmap='RdBu_r', norm=norm, linewidths=0, rasterized=True)
                A.set_xlabel(xl, fontsize=8.5, color=INK2); A.set_ylabel(yl, fontsize=8.5, color=INK2); A.set_aspect('equal' if vw == 'front' else 'auto')
                if vw in ('top', 'side'): A.set_xlim(F.AP.min() - 1, F.AP.max() + 1)
                ttl = f'{ttl} · {vt}'
            sub = '' if col.startswith('mu') else ''
            if col.startswith('mu'):
                sub = f'\nr with the eigenvector {np.corrcoef(T[col], T[e])[0, 1]:+.2f} · {100 * share[k][int(e[-1]) - 1]:.0f}% of its variance on it'
            A.set_title(ttl + sub, loc='left', fontsize=10, fontweight='bold', color=INK)
            for s_ in A.spines.values(): s_.set_visible(False)
            A.tick_params(length=0, labelsize=7.5, colors=INK2)
            cb = fig.colorbar(sc, ax=A, fraction=0.035, pad=0.01); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7, length=0)
    fig.suptitle('The main smooth spatial patterns of the 30 marmoset PFC regions (Moran eigenvectors of the region geometry) and the axes that follow them\n'
                 'atlas segments of Huang et al. 2026 coloured by the value of their region · left pair: flatmap of the authors, right pair: 3D view along the pattern · grey: other cortex of these sections',
                 x=0.01, ha='left', fontsize=12, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); p = c.pictures('spatial_eigenvectors_brain.png'); fig.savefig(p, dpi=140, facecolor='white'); plt.close(fig); print('saved', p)


if __name__ == '__main__':
    a = sys.argv; arg = lambda k, d: int(a[a.index(k) + 1]) if k in a else d
    eigen_maps() if '--eigen-maps' in a else run(arg('--nsets', 10000), arg('--nsurr', 10000))
