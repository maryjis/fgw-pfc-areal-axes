"""E5 of the validation plan: the rest of the snRNA-seq analysis.   python -m ottr.validation.e5_snrna
(a) PCA within Astro / Oligo / OPC (as snrna_types.pca, PFC nuclei, 3 000 HVG, 20 PCs): best |r| of a PC with the axis score per nucleus.
(b) gene-set null for the PCA comparison (excitatory subclasses and glia): NSETS random sets matched on expression decile, disjoint ends,
    scored per nucleus exactly as the axis score (sparse product, reference per sample), best |r| over the 20 PCs and R2 of the 20 PCs.
(c) development: subclass scores per sample at P0, P32, 3M, 7M (exports work/snrna_types_<age>) vs the atlas Pr-Al index of the
    regions a sample covers (snrna.references); all cortical samples of an age; plain permutation only (no spatial null outside PFC).
-> results/tables/validation/rows_snrna_pca.csv, e5_development.csv"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from ..config import cfg
from ..genescore import snrna_types as ST, snrna as SN, significance as SG
from . import common as C

NSETS = 2000; GROUPS = ST.EXC + ['Astro', 'Oligo', 'OPC']


def scores_matrix(Lg, smp, J):
    """J: sets x genes (columns of Lg) -> nuclei x sets module means, z per set by the sample-equal reference (snrna_types.reference)."""
    b, k = J.shape; I = sp.csr_matrix((np.full(J.size, 1.0 / k), (J.ravel(), np.repeat(np.arange(b), k))), shape=(Lg.shape[1], b)); M = np.asarray((Lg @ I).todense())
    u, ix = np.unique(smp, return_inverse=True); n = np.bincount(ix).astype(float); S = len(u); means = np.stack([M[ix == i].mean(0) for i in range(S)]); c = means.mean(0)
    sec = np.stack([((M[ix == i] - c) ** 2).mean(0) for i in range(S)]).mean(0); sd = np.sqrt(sec / (1 - (1 / (S * S * n)).sum())); return (M - c) / sd


def pca_null(L, cells, GI, rng):
    rows = []; tot = cells.n_count.values
    for g in GROUPS:
        q0, _, _, _ = ST.group_data(L, cells, g); lab0 = cells.region_label.values[q0]; q = q0[np.isin(lab0, ST.regions_ok(lab0))]; Lg = L[q].tocsr(); smp = cells.samplename.values[q]
        h = ST.hvg(Lg, 3000); D = ST.scaled(Lg, h); P, Z = ST.fit_pca(D, 20); Zs = (Z - Z.mean(0)) / Z.std(0); del D
        mg = np.asarray(Lg.mean(0)).ravel(); usable = np.flatnonzero(np.asarray((Lg > 0).mean(0)).ravel() >= 0.01); edges = np.quantile(mg[usable], np.linspace(0, 1, 11)[1:-1])
        by = {b: usable[np.digitize(mg[usable], edges) == b] for b in range(10)}; corr = lambda V: (Zs.T @ ((V - V.mean(0)) / V.std(0))) / len(V)
        for s in ST.SCORES:
            v = scores_matrix(Lg, smp, GI[s]['Pr'][None]) - scores_matrix(Lg, smp, GI[s]['Al'][None]); r = corr(v)[:, 0]; best, r2 = np.abs(r).max(), (r ** 2).sum()
            need = {e: dict(zip(*np.unique(np.digitize(mg[GI[s][e]], edges), return_counts=True))) for e in ('Pr', 'Al')}; nb, n2 = [], []
            for st in range(0, NSETS, 250):
                J, _ = ST.draw(rng, by, need, min(250, NSETS - st)); R = corr(scores_matrix(Lg, smp, J['Pr']) - scores_matrix(Lg, smp, J['Al'])); nb.append(np.abs(R).max(0)); n2.append((R ** 2).sum(0))
            nb, n2 = np.concatenate(nb), np.concatenate(n2); k = 1 if s == 'PrAl' else int(s[-1])
            rows.append(dict(domain='snRNA PCA within subclass', dataset=f'adult snRNA-seq, {g}', species='marmoset', axis='PrAl' if s == 'PrAl' else str(k), ours=f'{s} score per nucleus', feature='best |r| of PC1-20 with the score',
                             n=len(q), family='exploratory', spearman=float(best), p_extra=(1 + (nb >= best).sum()) / (NSETS + 1), p_extra_kind=f'gene-set null, best |r| over 20 PCs ({NSETS} sets)',
                             note=f'best PC {int(np.abs(r).argmax()) + 1}; R2 of 20 PCs {r2:.2f} (null median {np.median(n2):.2f}, p {(1 + (n2 >= r2).sum()) / (NSETS + 1):.4f}); per-nucleus r, not a region test'))
        print('pca null', g, flush=True)
    D = pd.DataFrame(rows); D['q_extra'] = D.groupby('dataset').p_extra.transform(lambda p: C.bh(p)); D.to_csv(os.path.join(C.out_dir(), 'rows_snrna_pca.csv'), index=False); return D


def development():
    ref = SN.references(); rows = []; rng = np.random.default_rng(3)
    for age in ('P0', 'P32', '3M', '7M'):
        X, genes, cells = SN.load(os.path.join(cfg().work, f'snrna_types_{age}')); X = X.astype(np.float32); L = ST.lognorm(X); GI = ST.gene_index(np.asarray(genes, str))
        for g in ST.GROUPS:
            q, Lg, smp, lab = ST.group_data(L, cells, g)
            if len(q) < 200: continue
            for s in ('PrAl', 'axis1', 'axis3'):
                v = 0
                for e, sg in (('Pr', 1), ('Al', -1)): m = ST.module(Lg, GI[s][e]); c0, sd = ST.reference(m, smp); v = v + sg * (m - c0) / sd
                T = pd.DataFrame(dict(v=v, s=smp, l=lab)).groupby(['s', 'l']).v.agg(['mean', 'size']).reset_index(); T = T[T['size'] >= 50]; y = T.l.map(ref['PrAl']).values.astype(float); f = np.isfinite(y)
                if f.sum() < 5: continue
                x = T['mean'].values[f]; r = SG.rho(x, y[f]); null = np.array([SG.rho(rng.permutation(x), y[f]) for _ in range(5000)])
                rows.append(dict(age=age, group=g, score=s, n_samples=int(f.sum()), rho_with_atlas_PrAl_index=r, p_perm=SG.p_two(null, r)))
        print('development', age, flush=True)
    D = pd.DataFrame(rows); D['q_perm'] = C.bh(D.p_perm); D.to_csv(os.path.join(C.out_dir(), 'e5_development.csv'), index=False); return D


if __name__ == '__main__':
    pd.set_option('display.width', 250); rng = np.random.default_rng(43)
    X, genes, cells = ST.load(); ST.check(X, genes, cells); L = ST.lognorm(X); GI = ST.gene_index(genes)
    N = pca_null(L, cells, GI, rng); print(N[['dataset', 'ours', 'spearman', 'p_extra', 'q_extra', 'note']].round(4).to_string(index=False))
    del X, L; Dv = development(); print(Dv.round(3).to_string(index=False))
