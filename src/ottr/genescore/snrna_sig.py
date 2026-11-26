"""Confidence intervals and significance of the region-order correlations of snrna_types (14 groups x 6 scores x 2 targets).
python -m ottr.genescore.snrna_sig [--gene]
CI   ci_bw      Spearman, Fisher z with the Bonett-Wright se sqrt((1 + rho^2/2)/(n-3)) (approximate at n = 6-7; |rho| = 1 clipped to 0.999)
     ci_boot    percentile bootstrap over regions (n = 6-7, discrete; resamples with < 4 different regions dropped)
     ci_nuclei  nuclei resampled within every region (measurement noise only)
p    p_perm exact region permutation (two-sided); q_perm BH over the 84 tests of a target; p_fwer Westfall-Young min-p (the same region
     permutation applied to all groups, so the dependence between groups that share samples is kept)
     p_msr / p_vario spatial null: surrogates of the target over the 30 regions (Moran / variogram, 3D geometry), averaged onto the
     snRNA regions like the target; BH per target. --gene: matched random gene sets (snrna_types.gene_null, 2000 sets)
-> results/tables/gene_axes/snrna_types/{significance,gene_null}.csv"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from ..config import cfg
from ..registry import axes_model
from . import significance as SG, snrna as SN, snrna_types as ST

NB, NS = 5000, 5000


def bh(p):
    return multipletests(p, method='fdr_bh')[1]


def run():
    O = ST.out_dir(); rng = np.random.default_rng(21); X, genes, cells = ST.load(); ST.check(X, genes, cells); L = ST.lognorm(X); GI = ST.gene_index(genes); T = ST.targets()
    # surrogates of the targets
    c = cfg(); R = pd.read_csv(c.tables('gene_axes/final_scores_regions.csv'), index_col=0); Z = axes_model('fgw2d_factors').npz
    mu = pd.DataFrame(Z['pb_mu'], index=Z['pb_regions']); sur = {}
    for s in ST.SCORES:
        k = 0 if s == 'PrAl' else int(s[-1]) - 1
        for t, v, wt in (('mu', mu[k], None), ('stereo', R[f'{s}_adultref'], R.n_bins)):
            regs = list(v.index); C = SG.marmoset_centroids(regs); f = np.isfinite(v.values) & np.isfinite(C).all(1); names = [r for r, q in zip(regs, f) if q]
            for m in ('msr', 'variogram'):
                S, _ = SG.surrogates(v.values[f].astype(float), C[f], NS, rng, m); S = pd.DataFrame(S, index=names); rows = []
                for lab, rr in SN.STEREO.items():
                    q = [r for r in rr if r in names]; w = np.ones(len(q)) if wt is None else wt.reindex(q).values; rows.append(np.average(S.loc[q].values, axis=0, weights=w))
                sur[s, t, m] = pd.DataFrame(np.array(rows), index=list(SN.STEREO))
    print('surrogates done', flush=True)
    out, vec = [], {}
    for g in ST.GROUPS:
        q, Lg, smp, lab = ST.group_data(L, cells, g); regs = ST.regions_ok(lab)
        for s in ST.SCORES:
            v = 0
            for e, sg in (('Pr', 1), ('Al', -1)): m = ST.module(Lg, GI[s][e]); ce, sd = ST.reference(m, smp); v = v + sg * (m - ce) / sd
            idx = [np.flatnonzero(lab == r) for r in regs]; x = np.array([v[w].mean() for w in idx])
            B = np.stack([np.array([v[rng.choice(w, len(w))].mean() for _ in range(1000)]) for w in idx])
            n = len(regs)
            for t in ('mu', 'stereo'):
                y = T[s, t].reindex(regs).values; o = ST.rho(x, y); zz = np.arctanh(np.clip(o, -.999, .999)); se = np.sqrt((1 + o ** 2 / 2) / (n - 3))
                bi = rng.integers(0, n, (NB, n)); bi = bi[[len(set(i)) >= 4 for i in bi]]; bs = np.array([ST.rho(x[i], y[i]) for i in bi]); bs = bs[np.isfinite(bs)]
                nu = SG.rank_cols(B).T @ SG.unit(y); d = dict(group=g, score=s, target=t, n_regions=n, n_nuclei=int(sum(map(len, idx))), rho=o, ci_bw_low=np.tanh(zz - 1.96 * se), ci_bw_high=np.tanh(zz + 1.96 * se),
                         ci_boot_low=np.quantile(bs, .025), ci_boot_high=np.quantile(bs, .975), ci_nuclei_low=np.quantile(nu, .025), ci_nuclei_high=np.quantile(nu, .975), p_perm=ST.perm_p(x, y))
                for m in ('msr', 'variogram'):
                    nl = SG.rank_cols(sur[s, t, m].loc[regs].values).T @ SG.unit(x); d['p_msr' if m == 'msr' else 'p_vario'] = SG.p_two(nl, o)
                out.append(d); vec[g, s, t] = (x, y, regs)
        print('sig', g, flush=True)
    D = pd.DataFrame(out)
    # Westfall-Young min-p: one permutation of the 7 region labels applied to every test
    perms = np.array(list(itertools.permutations(range(7)))); pos = {r: i for i, r in enumerate(ST.REGIONS)}
    for t in ('mu', 'stereo'):
        keys = [(g, s) for g in ST.GROUPS for s in ST.SCORES]; P = np.empty((len(perms), len(keys)))
        for j, (g, s) in enumerate(keys):
            x, y, regs = vec[g, s, t]; full = T[s, t].reindex(ST.REGIONS).values; ii = [pos[r] for r in regs]; ry = SG.rank_cols(full[perms][:, ii].T); r_ = np.abs(ry.T @ SG.unit(x))
            P[:, j] = [(r_ >= a - 1e-12).mean() for a in r_]
        minp = P.min(1); obs = P[0]; assert (perms[0] == np.arange(7)).all()
        for j, (g, s) in enumerate(keys): D.loc[(D.group == g) & (D.score == s) & (D.target == t), 'p_fwer'] = (minp <= obs[j] + 1e-12).mean()
    for p in ('p_perm', 'p_msr', 'p_vario'):
        D['q' + p[1:]] = np.nan
        for t in ('mu', 'stereo'): w = D.target == t; D.loc[w, 'q' + p[1:]] = bh(D.loc[w, p])
    D.to_csv(os.path.join(O, 'significance.csv'), index=False); print('significance done', flush=True)
    if '--gene' in sys.argv:
        N = ST.gene_null(L, cells, GI, T, genes, 2000)
        for p in ('p_two', 'p_one'): ST.bh(N, p, ['target', 'pool'])
        N.to_csv(os.path.join(O, 'gene_null.csv'), index=False); print('gene null done', flush=True)


if __name__ == '__main__': run()
