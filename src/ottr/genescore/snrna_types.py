"""Which cell subclasses reproduce the areal axes: the gene-set scores (PrAl, axes 1-5) within every subclass of the adult snRNA-seq.
python -m ottr.genescore.snrna_types [--sets 10000] [--reps 100]     (after scripts/R/snrna_export.R <All object> <8f> work/snrna_types adult)

Unit of analysis = adult PFC sample (7 regions, one sample each); a region counts for a group when it has >= MIN_N nuclei.
scores()       per nucleus as ottr.genescore.snrna (demo formula, z within the group, reference = all adult samples of the group),
               mean per sample; Spearman over regions with (a) the Stereo-seq score of the same regions (snrna.STEREO, adult reference)
               and (b) mu of the axis; exact permutation p, leave-one-out, without A32, without A24A32
split_half()   nuclei of every region split in two halves: Spearman of the two region orders (sampling noise of nuclei only)
matched()      the same number of nuclei per region and the same UMI per nucleus in every group (thinning), repeated
gene_null()    random gene sets of the same size, matched on the expression decile within the group, disjoint ends, scored with
               exactly the formula of scores() (variance of a set mean from the gene x gene second moments)
composition()  region differences of the pooled tissue split into expression (common subclass proportions) and composition
contributions() per gene of an axis: covariance of its region means with the target in every group; agreement between groups
               against the split-half agreement within a group
pca()          PCA of the nuclei within one excitatory subclass (no axis genes selected): PCs against the axis scores per nucleus, region means
               against mu / Stereo-seq; controls: UMI, region share of variance, split-half of the PCs, random gene sets, without axis genes
Caveats: one sample per region, adult donor not given, so nothing here measures variation between animals.
-> results/tables/gene_axes/snrna_types/*.csv"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata
from ..config import cfg, ensure
from ..registry import axes_model
from . import pral, snrna as SN

SCORES = SN.SCORES; MIN_N = 100; N_HALF = 200; TARGET_UMI = 3000
REGIONS = list(SN.STEREO)
GROUPS = {'L2/3 IT + ExN_dev': ['L2/3 IT', 'ExN_dev_1', 'ExN_dev_2'], 'L2/3 IT (no ExN_dev)': ['L2/3 IT'], 'L4 IT': ['L4 IT', 'L4 IT-V1'],
          'L5 IT': ['L5 IT'], 'L6 IT': ['L6 IT'], 'L6 CT': ['L6 CT'], 'L5 ET': ['L5 ET'], 'Pvalb': ['Pvalb'], 'Sst': ['Sst'],
          'Vip': ['Vip'], 'Lamp5': ['Lamp5'], 'Astro': ['Astro'], 'Oligo': ['Oligo'], 'OPC': ['OPC']}


def rho(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def out_dir():
    return ensure(cfg().tables('gene_axes/snrna_types'))


def load():
    X, genes, cells = SN.load(os.path.join(cfg().work, 'snrna_types'))
    return X.astype(np.float32), np.asarray(genes, str), cells.reset_index(drop=True)


def targets():
    """{(score, target): Series over REGIONS}: the Stereo-seq score (bin-weighted) and mu, as snrna.compare_stereo / COVERS."""
    c = cfg(); R = pd.read_csv(c.tables('gene_axes/final_scores_regions.csv'), index_col=0)
    Z = axes_model('fgw2d_factors').npz; mu = pd.DataFrame(Z['pb_mu'], index=Z['pb_regions']); T = {}
    for s in SCORES:
        k = 0 if s == 'PrAl' else int(s[-1]) - 1; a, b = {}, {}
        for lab, regs in SN.STEREO.items():
            q = [r for r in regs if r in R.index]; a[lab] = np.average(R.loc[q, f'{s}_adultref'], weights=R.loc[q, 'n_bins'])
            b[lab] = mu[k].reindex(regs).dropna().mean()
        T[s, 'stereo'] = pd.Series(a); T[s, 'mu'] = pd.Series(b)
    return T


def lognorm(X):
    t = np.asarray(X.sum(1)).ravel(); L = sp.csr_matrix(X, dtype=np.float64).multiply((1e4 / t)[:, None]).tocsr(); L.data = np.log1p(L.data); return L


def gene_index(genes):
    pos = {g: i for i, g in enumerate(genes)}; S = SN.gene_sets()
    return {s: {e: np.array([pos[g] for g in S[s][e] if g in pos]) for e in ('Pr', 'Al')} for s in SCORES}


def reference(v, sample):
    """centre and scale of a module mean over nuclei: equal samples, sd corrected by 1 - sum(w^2) (pral.fit_reference, one age)."""
    u, ix = np.unique(sample, return_inverse=True); n = np.bincount(ix).astype(float); S = len(u)
    m = np.bincount(ix, v) / n; c = m.mean(); sec = np.bincount(ix, (v - c) ** 2) / n
    return c, np.sqrt(sec.mean() / (1 - (1 / (S * S * n)).sum()))


def regions_ok(lab, minimum=MIN_N):
    n = pd.Series(lab).value_counts(); return [r for r in REGIONS if n.get(r, 0) >= minimum]


def perm_p(x, y):
    n = len(x); P = np.array(list(itertools.permutations(range(n)))); rx, ry = rankdata(x), rankdata(y); rx, ry = rx - rx.mean(), ry - ry.mean()
    null = (rx[P] @ ry) / np.sqrt((rx @ rx) * (ry @ ry)); return float((np.abs(null) >= abs(rho(x, y)) - 1e-12).mean())


def group_data(L, cells, g):
    q = np.flatnonzero(cells.subclass.isin(GROUPS[g]).values); smp = cells.samplename.values[q]; n = pd.Series(smp).value_counts()
    q = q[n.reindex(smp).values >= 2]; return q, L[q], cells.samplename.values[q], cells.region_label.values[q]


def module(L, idx):
    return np.asarray(L[:, idx].sum(1)).ravel() / len(idx)


def region_mean(v, lab, regs):
    return np.array([v[lab == r].mean() for r in regs])


# ---------------------------------------------------------------- scores
def scores(L, cells, GI, T, rng):
    rows, smp_rows, cell_scores = [], [], {}
    for g in GROUPS:
        q, Lg, smp, lab = group_data(L, cells, g); regs = regions_ok(lab)
        for s in SCORES:
            v = 0
            for e, sg in (('Pr', 1), ('Al', -1)):
                m = module(Lg, GI[s][e]); c, sd = reference(m, smp); v = v + sg * (m - c) / sd
            cell_scores[g, s] = (v, lab)
            for r in pd.unique(lab): smp_rows.append(dict(group=g, score=s, region_label=r, mean=v[lab == r].mean(), n_nuclei=int((lab == r).sum())))
            x = region_mean(v, lab, regs)
            half = []
            for _ in range(N_HALF):
                a, b = [], []
                for r in regs:
                    w = rng.permutation(np.flatnonzero(lab == r)); h = len(w) // 2; a.append(v[w[:h]].mean()); b.append(v[w[h:]].mean())
                half.append(rho(a, b))
            for tn in ('stereo', 'mu'):
                y = T[s, tn].reindex(regs).values; loo = [rho(np.delete(x, i), np.delete(y, i)) for i in range(len(regs))]
                d = dict(group=g, score=s, target=tn, n_regions=len(regs), missing=','.join(r for r in REGIONS if r not in regs), n_nuclei=int(np.isin(lab, regs).sum()),
                         rho=rho(x, y), pearson=float(np.corrcoef(x, y)[0, 1]), p_perm=perm_p(x, y), loo_min=min(loo), loo_max=max(loo),
                         spread=float(x.max() - x.min()), split_half=float(np.median(half)), split_half_q05=float(np.quantile(half, 0.05)))
                for drop in ('A32', 'A24A32'):
                    k = [i for i, r in enumerate(regs) if r != drop]; d[f'rho_no_{drop}'] = rho(x[k], y[k]); d[f'p_no_{drop}'] = perm_p(x[k], y[k])
                rows.append(d)
        print('scores', g, len(q), 'nuclei', len(regs), 'regions', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(smp_rows)


# ---------------------------------------------------------------- equal nuclei and UMI
def matched(X, cells, GI, T, reps, seed=1):
    rng = np.random.default_rng(seed); out = []; tot = np.asarray(X.sum(1)).ravel()
    for n_per in (200, 100):
        for g in GROUPS:
            ok = cells.subclass.isin(GROUPS[g]).values & (tot >= TARGET_UMI); lab = cells.region_label.values
            pool = {r: np.flatnonzero(ok & (lab == r)) for r in REGIONS}; regs = [r for r in REGIONS if len(pool[r]) >= n_per]
            if len(regs) < 6: continue
            acc = {(s, t): [] for s in SCORES for t in ('stereo', 'mu')}
            for _ in range(reps):
                ix = np.concatenate([rng.choice(pool[r], n_per, replace=False) for r in regs]); rl = np.repeat(regs, n_per); Y = X[ix].tocsr().astype(np.float64)
                p = TARGET_UMI / np.asarray(Y.sum(1)).ravel(); Y.data = rng.binomial(np.round(Y.data).astype(np.int64), np.minimum(np.repeat(p, np.diff(Y.indptr)), 1)).astype(np.float64)
                Ln = lognorm(Y)
                for s in SCORES:
                    v = 0
                    for e, sg in (('Pr', 1), ('Al', -1)):
                        m = module(Ln, GI[s][e]); c, sd = reference(m, rl); v = v + sg * (m - c) / sd
                    x = region_mean(v, rl, regs)
                    for t in ('stereo', 'mu'): acc[s, t].append(rho(x, T[s, t].reindex(regs).values))
            for (s, t), a in acc.items():
                out.append(dict(group=g, score=s, target=t, nuclei_per_region=n_per, umi=TARGET_UMI, n_regions=len(regs), reps=reps, rho_median=np.median(a),
                                rho_q05=np.quantile(a, .05), rho_q95=np.quantile(a, .95)))
            print('matched', n_per, g, len(regs), flush=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- gene null
def draw(rng, by_bin, need, b):
    """b draws without replacement within a bin for both ends jointly (disjoint); returns {'Pr': b x n, 'Al': b x n}, flag of replacement."""
    out = {'Pr': [], 'Al': []}; repl = False
    for k, pool in by_bin.items():
        a, c = need['Pr'].get(k, 0), need['Al'].get(k, 0); m = a + c
        if m == 0: continue
        if len(pool) >= m: J = pool[np.argsort(rng.random((b, len(pool))), axis=1)[:, :m]]
        else: J = rng.choice(pool, (b, m)); repl = True
        out['Pr'].append(J[:, :a]); out['Al'].append(J[:, a:])
    return {e: np.concatenate(v, 1) for e, v in out.items()}, repl


def null_group(Lg, smp, lab, regs, GI, T, pools, nsets, seed):
    rng = np.random.default_rng(seed); u, ix = np.unique(smp, return_inverse=True); n = np.bincount(ix).astype(float); S = len(u)
    mean_g = np.asarray(Lg.mean(0)).ravel(); use = np.flatnonzero(mean_g > 0); loc = -np.ones(Lg.shape[1], int); loc[use] = np.arange(len(use))
    D = Lg[:, use].toarray().astype(np.float32); w = (1 / (S * n[ix])).astype(np.float32)              # equal samples
    C2 = (D * w[:, None]).T @ D                                                                        # weighted second moments
    mu_g = (D * w[:, None]).sum(0, dtype=np.float64); corr = 1 - (1 / (S * S * n)).sum()
    M = np.stack([D[lab == r].mean(0, dtype=np.float64) for r in regs]); del D
    def zscore(idx):                                                                                   # sets x genes (local) -> regions x sets
        k = idx.shape[1]; c = mu_g[idx].mean(1); sec = np.array([C2[np.ix_(i, i)].sum(dtype=np.float64) for i in idx]) / k ** 2
        sd = np.sqrt(np.maximum(sec - c ** 2, 1e-12) / corr); return (M[:, idx].mean(2) - c) / sd
    out = []; gm = mean_g[use]
    for s in SCORES:
        ends = {e: loc[GI[s][e]] for e in ('Pr', 'Al')}; ends = {e: v[v >= 0] for e, v in ends.items()}
        # genes never detected in the group stay in the score's denominator: constant factor, no effect on z
        obs = (zscore(ends['Pr'][None]) - zscore(ends['Al'][None]))[:, 0]
        for pn, pool in pools.items():
            cand = np.array(sorted(set(loc[pool][loc[pool] >= 0]) | set(ends['Pr']) | set(ends['Al'])))
            edges = np.quantile(np.log10(gm[cand]), np.linspace(0, 1, 11)[1:-1]); bn = lambda i: np.digitize(np.log10(gm[i]), edges)
            by = {k: cand[bn(cand) == k] for k in range(10)}; need = {e: dict(zip(*np.unique(bn(v), return_counts=True))) for e, v in ends.items()}
            null, rp = [], False
            for st in range(0, nsets, 250):
                J, r_ = draw(rng, by, need, min(250, nsets - st)); rp |= r_; null.append(zscore(J['Pr']) - zscore(J['Al']))
            null = np.concatenate(null, 1); R0 = np.apply_along_axis(rankdata, 0, null); R0 = R0 - R0.mean(0); R0 /= np.linalg.norm(R0, axis=0)
            for t in ('stereo', 'mu'):
                y = T[s, t].reindex(regs).values; ry = rankdata(y); ry = (ry - ry.mean()) / np.linalg.norm(ry - ry.mean()); nr = ry @ R0; o = rho(obs, y)
                yc = (y - y.mean()) / np.linalg.norm(y - y.mean()); nc = null - null.mean(0); np_ = (yc @ nc) / np.linalg.norm(nc, axis=0); op = float(np.corrcoef(obs, y)[0, 1])
                slope = yc @ nc; oslope = float(yc @ (obs - obs.mean()))
                out.append(dict(score=s, target=t, pool=pn, n_pool=len(cand), n_pos=len(ends['Pr']), n_neg=len(ends['Al']), nsets=nsets, with_replacement=rp, rho=o,
                                null_mean=nr.mean(), null_sd=nr.std(), null_abs_q95=np.quantile(np.abs(nr), .95), z=(o - nr.mean()) / nr.std(),
                                p_two=(1 + (np.abs(nr) >= abs(o) - 1e-12).sum()) / (nsets + 1), p_one=(1 + (nr >= o - 1e-12).sum()) / (nsets + 1),
                                pearson=op, p_one_pearson=(1 + (np_ >= op - 1e-12).sum()) / (nsets + 1),
                                z_slope=(oslope - slope.mean()) / slope.std(), p_one_slope=(1 + (slope >= oslope - 1e-12).sum()) / (nsets + 1)))
    return out


def _null_job(job):
    from threadpoolctl import threadpool_limits
    j, g = job; L, cells, GI, T, pools, nsets = _SHARED
    with threadpool_limits(8):
        q, Lg, smp, lab = group_data(L, cells, g); regs = regions_ok(lab)
        r = [dict(group=g, n_regions=len(regs), **d) for d in null_group(Lg, smp, lab, regs, GI, T, pools, nsets, 100 + j)]
    print('gene null', g, flush=True); return r


def gene_null(L, cells, GI, T, genes, nsets):
    c = cfg(); up = {g.upper(): i for i, g in enumerate(genes)}
    model = pd.read_csv(os.path.join(c.work, 'genescore', 'encoder_axes.csv'), index_col=0).index.str.upper()
    pools = {'genome': np.arange(len(genes)), 'model': np.array([up[g] for g in model if g in up])}; out = []
    global _SHARED; _SHARED = (L, cells, GI, T, pools, nsets)
    import multiprocessing as mp
    with mp.get_context('fork').Pool(len(GROUPS)) as pool:                      # one process per group, data shared by fork
        for d in pool.imap(_null_job, list(enumerate(GROUPS)), chunksize=1): out += d
    return pd.DataFrame(out)


def bh(df, p, by):
    from statsmodels.stats.multitest import multipletests
    q = 'q' + p[1:]; df[q] = np.nan
    for _, i in df.groupby(by).groups.items(): df.loc[i, q] = multipletests(df.loc[i, p], method='fdr_bh')[1]
    return df


# ---------------------------------------------------------------- expression or composition
def composition(L, cells, GI, T, rng):
    rows, prop_rows = [], []; lab = cells.region_label.values; cl = {'neurons': cells['class_'] != 'NNC', 'ExN': cells['class_'] == 'ExN', 'all nuclei': cells['class_'] != ''}
    for uni, m in cl.items():
        q = np.flatnonzero(m.values & np.isin(lab, REGIONS)); sub = cells.subclass.values[q]; rl = lab[q]
        N = pd.crosstab(rl, sub).reindex(REGIONS); keep = [s for s in N.columns if (N[s] >= 20).all()]; N = N[keep]; P = N.div(N.sum(1), axis=0); W = P.mean(0).values
        kq = np.isin(sub, keep); q, sub, rl = q[kq], sub[kq], rl[kq]
        for s in SCORES:
            v = 0
            for e, sg in (('Pr', 1), ('Al', -1)):
                mm = module(L[q], GI[s][e]); c, sd = reference(mm, rl); v = v + sg * (mm - c) / sd
            B = pd.DataFrame(dict(v=v, r=rl, s=sub)).groupby(['r', 's']).v.mean().unstack().reindex(REGIONS)[keep].values
            parts = {'natural': (P.values * B).sum(1), 'expression (fixed proportions)': (W * B).sum(1), 'composition only': (P.values * B.mean(0)).sum(1)}
            for t in ('stereo', 'mu'):
                y = T[s, t].reindex(REGIONS).values
                for nm, x in parts.items(): rows.append(dict(universe=uni, n_subclasses=len(keep), excluded=','.join(k for k in pd.unique(cells.subclass[m]) if k not in keep),
                                                             score=s, target=t, part=nm, rho=rho(x, y), pearson=float(np.corrcoef(x, y)[0, 1]), spread=float(np.ptp(x))))
        if uni == 'neurons':
            for s in SCORES:
                for t in ('stereo', 'mu'):
                    y = T[s, t].reindex(REGIONS).values
                    for k in keep: prop_rows.append(dict(subclass=k, score=s, target=t, rho=rho(P[k].values, y), p_perm=perm_p(P[k].values, y), prop_min=P[k].min(), prop_max=P[k].max()))
    return pd.DataFrame(rows), pd.DataFrame(prop_rows)


# ---------------------------------------------------------------- contribution of the genes
def contributions(L, cells, GI, T, genes, rng, splits=20):
    rows, agree = [], []
    for s in SCORES:
        idx = np.concatenate([GI[s]['Pr'], GI[s]['Al']]); sg = np.r_[np.ones(len(GI[s]['Pr'])), -np.ones(len(GI[s]['Al']))]; K, H, det = {}, {}, {}
        regs_all = [r for r in REGIONS]
        for g in GROUPS:
            q, Lg, smp, lab = group_data(L, cells, g); regs = regions_ok(lab); y = T[s, 'stereo'].reindex(regs).values; y = (y - y.mean()) / y.std()
            A = Lg[:, idx].toarray(); det[g] = (A > 0).mean(0); con = lambda rowsets: sg * (np.array([A[w].mean(0) for w in rowsets]).T @ y) / len(y)
            full = [np.flatnonzero(lab == r) for r in regs]; K[g] = con(full); hs = []
            for _ in range(splits):
                pm = [rng.permutation(w) for w in full]; a = con([w[:len(w) // 2] for w in pm]); b = con([w[len(w) // 2:] for w in pm]); hs.append(rho(a, b))
            H[g] = float(np.median(hs))
            for i, gi in enumerate(idx): rows.append(dict(score=s, group=g, gene=genes[gi], end='positive' if sg[i] > 0 else 'negative', contribution=K[g][i], detected=det[g][i], n_regions=len(regs)))
        for a, b in itertools.combinations(GROUPS, 2):
            top = lambda g: set(np.argsort(-K[g])[:50]); agree.append(dict(score=s, group_a=a, group_b=b, rho=rho(K[a], K[b]), split_half_a=H[a], split_half_b=H[b],
                                                                          rho_over_ceiling=rho(K[a], K[b]) / np.sqrt(max(H[a], 1e-3) * max(H[b], 1e-3)), top50_shared=len(top(a) & top(b)), n_genes=len(idx)))
        print('contributions', s, flush=True)
    return pd.DataFrame(rows), pd.DataFrame(agree)


# ---------------------------------------------------------------- PCA within a subclass
EXC = ['L2/3 IT + ExN_dev', 'L2/3 IT (no ExN_dev)', 'L4 IT', 'L5 IT', 'L6 IT', 'L6 CT', 'L5 ET']


def hvg(Lg, n, exclude=()):
    m = np.asarray(Lg.mean(0)).ravel(); v = np.asarray(Lg.multiply(Lg).mean(0)).ravel() - m ** 2; det = np.asarray((Lg > 0).mean(0)).ravel(); ok = det >= 0.01
    ok[list(exclude)] = False; i = np.flatnonzero(ok); b = np.digitize(m[i], np.quantile(m[i], np.linspace(0, 1, 21)[1:-1])); z = np.zeros(len(i))
    for k in np.unique(b): w = b == k; d = np.log(v[i][w] / m[i][w]); z[w] = (d - d.mean()) / max(d.std(), 1e-9)      # dispersion within mean bins
    return i[np.argsort(-z)[:n]]


def fit_pca(D, k, seed=0):
    from sklearn.decomposition import PCA
    p = PCA(k, svd_solver='randomized', random_state=seed).fit(D); return p, p.transform(D)


def scaled(Lg, h):
    D = Lg[:, h].toarray(); sd = D.std(0); sd[sd == 0] = 1; return np.clip((D - D.mean(0)) / sd, -10, 10)


def pca(L, cells, GI, T, n_hvg=3000, k=20, n_null=0, seed=3):
    rng = np.random.default_rng(seed); rows, pcs, r2 = [], [], []; axis_genes = np.unique(np.concatenate([v for d in GI.values() for v in d.values()])); tot = cells.n_count.values
    for g in EXC:
        q0, _, _, _ = group_data(L, cells, g)
        for uni, regsel in (('PFC', REGIONS), ('all 13 samples', None)):
            q = q0 if regsel is None else q0[np.isin(cells.region_label.values[q0], regions_ok(cells.region_label.values[q0]))]
            Lg = L[q]; lab = cells.region_label.values[q]; smp = cells.samplename.values[q]; lu = np.log(tot[q]); regs = regions_ok(lab)
            mg = np.asarray(Lg.mean(0)).ravel(); usable = np.flatnonzero(np.asarray((Lg > 0).mean(0)).ravel() >= 0.01); edges = np.quantile(mg[usable], np.linspace(0, 1, 11)[1:-1])
            def score(pos, neg):
                v = 0
                for idx, sg in ((pos, 1), (neg, -1)): m = module(Lg, idx); c, sd = reference(m, smp); v = v + sg * (m - c) / sd
                return v
            V = {s: score(GI[s]['Pr'], GI[s]['Al']) for s in SCORES}
            if n_null: by = {b: usable[np.digitize(mg[usable], edges) == b] for b in range(10)}; null = {s: [] for s in SCORES}
            for s in SCORES if n_null else []:
                need = {e: dict(zip(*np.unique(np.digitize(mg[GI[s][e]], edges), return_counts=True))) for e in ('Pr', 'Al')}; J, _ = draw(rng, by, need, n_null)
                null[s] = np.array([score(J['Pr'][i], J['Al'][i]) for i in range(n_null)])
            for var, exc in (('all genes', ()), ('without axis genes', axis_genes)):
                h = hvg(Lg, n_hvg, exc); D = scaled(Lg, h); P, Z = fit_pca(D, k); w = rng.permutation(len(q)); ha, hb = w[:len(w) // 2], w[len(w) // 2:]
                Ca, Cb = fit_pca(D[ha], k)[0].components_, fit_pca(D[hb], k)[0].components_; rep = np.abs(Ca[:, :] @ P.components_.T).max(0), np.abs(Cb @ P.components_.T).max(0)
                Zs = (Z - Z.mean(0)) / Z.std(0); eta = []
                for j in range(k): mr = pd.Series(Z[:, j]).groupby(lab).agg(['mean', 'size']); eta.append(float((mr['size'] * (mr['mean'] - Z[:, j].mean()) ** 2).sum() / ((Z[:, j] - Z[:, j].mean()) ** 2).sum()))
                for j in range(k):
                    pcs.append(dict(group=g, universe=uni, variant=var, pc=j + 1, n_nuclei=len(q), n_hvg=len(h), explained=P.explained_variance_ratio_[j], r_log_umi=np.corrcoef(Z[:, j], lu)[0, 1],
                                    eta2_region=eta[j], repro_a=rep[0][j], repro_b=rep[1][j]))
                corr = lambda v: (Zs.T @ ((v - v.mean()) / v.std())) / len(v)
                for s in SCORES:
                    r = corr(V[s]); d2 = dict(group=g, universe=uni, variant=var, score=s, n_nuclei=len(q), r2_20pc=float((r ** 2).sum()), best_pc=int(np.argmax(np.abs(r))) + 1, best_abs_r=float(np.abs(r).max()))
                    if n_null:
                        nz = np.array([corr(v) for v in null[s]]); n2 = (nz ** 2).sum(1)
                        d2.update(null_r2_median=float(np.median(n2)), p_r2=(1 + (n2 >= d2['r2_20pc']).sum()) / (n_null + 1), p_best=(1 + (np.abs(nz).max(1) >= d2['best_abs_r']).sum()) / (n_null + 1))
                    r2.append(d2)
                    for j in range(k):
                        x = region_mean(Z[:, j], lab, regs); d = dict(group=g, universe=uni, variant=var, score=s, pc=j + 1, r_nuclei=r[j])
                        for t in ('stereo', 'mu'): d[f'rho_region_{t}'] = rho(x, T[s, t].reindex(regs).values)
                        rows.append(d)
        print('pca', g, flush=True)
    return pd.DataFrame(rows), pd.DataFrame(pcs), pd.DataFrame(r2)


def check(X, genes, cells):
    """Labels against the final metadata; L2/3 and L4 against the earlier export and scores."""
    f = pd.read_csv('/workspace/preprocess_ctx/data/8f_metarefine_all_metadata.csv', low_memory=False).set_index('seurat_cell_id').loc[cells.cell]
    for a in ('samplename', 'age', 'subclass'): assert (f[a].values == cells[a].values).all(), a
    cells['class_'] = f['class'].values
    old = pd.read_csv(os.path.join(cfg().work, 'snrna', 'cells.tsv'), sep='\t', usecols=['cell', 'age', 'n_count']); old = old[old.age == 'adult'].set_index('cell').n_count
    new = cells.set_index('cell').n_count.reindex(old.index); assert (new.values == old.values).all()
    print('labels match 8f for', len(cells), 'nuclei; n_count equals the earlier export for', len(old), flush=True)


def verify(smp, T):
    A = pd.read_csv(cfg().tables('gene_axes/snrna/scores_samples.csv')); A = A[A.age == 'adult']; d = []
    for g_new, g_old in (('L2/3 IT + ExN_dev', 'L2/3 IT'), ('L4 IT', 'L4 IT')):
        for s in SCORES:
            a = A[A.group == g_old].set_index('region_label')[f'{s}|adultref']; b = smp[(smp.group == g_new) & (smp.score == s)].set_index('region_label')['mean'].reindex(a.index)
            d.append(np.abs(a - b).max())
    print('max |new - earlier sample score| L2/3, L4:', max(d), flush=True); return max(d)


def run(nsets=10000, reps=100):
    O = out_dir(); rng = np.random.default_rng(0); X, genes, cells = load(); check(X, genes, cells); L = lognorm(X); GI = gene_index(genes); T = targets()
    R, smp = scores(L, cells, GI, T, rng); bh(R, 'p_perm', ['target']); verify(smp, T)
    R.to_csv(os.path.join(O, 'scores.csv'), index=False); smp.to_csv(os.path.join(O, 'scores_samples.csv'), index=False)
    C, P = composition(L, cells, GI, T, rng); C.to_csv(os.path.join(O, 'composition.csv'), index=False); P.to_csv(os.path.join(O, 'proportions.csv'), index=False); print('composition done', flush=True)
    K, A = contributions(L, cells, GI, T, genes, rng); K.to_csv(os.path.join(O, 'contributions.csv.gz'), index=False); A.to_csv(os.path.join(O, 'contribution_agreement.csv'), index=False)
    Q, PC, R2 = pca(L, cells, GI, T); Q.to_csv(os.path.join(O, 'pca_axes.csv'), index=False); PC.to_csv(os.path.join(O, 'pca_components.csv'), index=False); R2.to_csv(os.path.join(O, 'pca_r2.csv'), index=False)
    M = matched(X, cells, GI, T, reps); M.to_csv(os.path.join(O, 'matched.csv'), index=False)
    N = gene_null(L, cells, GI, T, genes, nsets); bh(N, 'p_one', ['target', 'pool']); bh(N, 'p_one_slope', ['target', 'pool']); N.to_csv(os.path.join(O, 'gene_null.csv'), index=False)
    print('done', flush=True)


def run_pca():
    O = out_dir(); X, genes, cells = load(); check(X, genes, cells); L = lognorm(X); Q, PC, R2 = pca(L, cells, gene_index(genes), targets())
    Q.to_csv(os.path.join(O, 'pca_axes.csv'), index=False); PC.to_csv(os.path.join(O, 'pca_components.csv'), index=False); R2.to_csv(os.path.join(O, 'pca_r2.csv'), index=False); print('pca done', flush=True)


if __name__ == '__main__':
    a = sys.argv
    if '--pca' in a: run_pca(); sys.exit()
    run(int(a[a.index('--sets') + 1]) if '--sets' in a else 10000, int(a[a.index('--reps') + 1]) if '--reps' in a else 100)
