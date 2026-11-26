"""PAD (Pr-Al gene difference, Huang et al. 2026) of our gene sets within subclasses of the snRNA-seq.
python -m ottr.genescore.pad
PAD = mean(gene-gene r within the + set and within the - set) - mean(r between the sets); Pearson over nuclei, log1p CP10k,
diagonal excluded, genes without variance dropped. Uses block sums: sum of r over a block = |sum of z-scored genes|^2 / n.
Variants: all nuclei / matched (N_REG nuclei per region, UMI thinned to UMI, REPS draws, median); each pooled and with the region
(sample) mean of every gene subtracted ('within region'). Null: random sets matched on the expression decile, disjoint ends.
-> results/tables/gene_axes/snrna_types/pad.csv (adult, 14 groups), pad_ages.csv (L2/3, L4; five ages)"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from ..config import cfg
from . import snrna as SN, snrna_types as ST

UMI, N_REG, REPS, NNULL = 2000, 80, 20, 1000


def zscore(D, lab=None):
    if lab is not None:
        D = D.copy()
        for r in pd.unique(lab): w = lab == r; D[w] -= D[w].mean(0)
    sd = D.std(0); ok = sd > 1e-8; Z = np.zeros_like(D); Z[:, ok] = (D[:, ok] - D[:, ok].mean(0)) / sd[ok]; return Z, ok


def pad(Z, ok, A, B):
    """A, B: sets x genes (column indices). PAD of every set pair."""
    n = len(Z); out = []
    for s in range(0, len(A), 100):
        a, b = A[s:s + 100], B[s:s + 100]; m = lambda J: sp.csr_matrix((np.ones(J.size, np.float32), J.ravel(), np.arange(0, J.size + 1, J.shape[1])), shape=(len(J), Z.shape[1]))
        Sa, Sb = np.asarray((m(a) @ Z.T)), np.asarray((m(b) @ Z.T)); ka, kb = ok[a].sum(1).astype(float), ok[b].sum(1).astype(float)
        Sa, Sb = Sa.astype(np.float64), Sb.astype(np.float64)
        paa = ((Sa ** 2).sum(1) / n - ka) / (ka * (ka - 1)); pbb = ((Sb ** 2).sum(1) / n - kb) / (kb * (kb - 1)); pab = (Sa * Sb).sum(1) / n / (ka * kb)
        out.append(np.c_[(paa + pbb) / 2 - pab, (paa + pbb) / 2, pab])
    return np.concatenate(out)


def null_sets(mean, cols, GI, rng, n):
    """cols: gene index -> column; matched sets for every score."""
    edges = np.quantile(mean, np.linspace(0, 1, 11)[1:-1]); bn = np.digitize(mean, edges); by = {b: np.flatnonzero(bn == b) for b in range(10)}; out = {}
    for s in SN.SCORES:
        e = {k: cols[v][cols[v] >= 0] for k, v in GI[s].items()}; need = {k: dict(zip(*np.unique(bn[v], return_counts=True))) for k, v in e.items()}
        J, _ = ST.draw(rng, by, need, n); out[s] = (e, J)
    return out


def columns(Lq, GI):
    det = np.asarray((Lq > 0).mean(0)).ravel(); ax = np.unique(np.concatenate([v for d in GI.values() for v in d.values()])); use = det >= 0.01; use[ax] = det[ax] > 0
    ix = np.flatnonzero(use); cols = -np.ones(len(det), int); cols[ix] = np.arange(len(ix)); return ix, cols, np.asarray(Lq.mean(0)).ravel()[ix]


def one(X, L, q, lab, GI, rng, tag):
    rows = []; ix, cols, mean = columns(L[q], GI); sets = null_sets(mean, cols, GI, rng, NNULL); D = L[q][:, ix].toarray().astype(np.float32); lq = lab[q]
    for within in (False, True):
        Z, ok = zscore(D, lq if within else None)
        for s, (e, J) in sets.items():
            o = pad(Z, ok, e['Pr'][None], e['Al'][None])[0]; n = pad(Z, ok, J['Pr'], J['Al'])[:, 0]
            rows.append(dict(tag, score=s, variant='all nuclei', within_region=within, n_nuclei=len(q), n_regions=len(pd.unique(lq)), pad=o[0], within_sets=o[1], between_sets=o[2],
                             null_mean=n.mean(), null_sd=n.std(), z=(o[0] - n.mean()) / n.std(), p=(1 + (n >= o[0]).sum()) / (len(n) + 1)))
    del D
    tot = np.asarray(X[q].sum(1)).ravel(); elig = q[tot >= UMI]; cnt = pd.Series(lab[elig]).value_counts(); regs = [r for r in cnt.index if cnt[r] >= N_REG]
    if len(regs) >= 4:
        acc = {}
        for _ in range(REPS):
            pk = np.concatenate([rng.choice(elig[lab[elig] == r], N_REG, replace=False) for r in regs]); Y = X[pk].tocsr().astype(np.float64); t = np.asarray(Y.sum(1)).ravel()
            Y.data = rng.binomial(np.round(Y.data).astype(np.int64), np.repeat(np.minimum(UMI / t, 1), np.diff(Y.indptr))).astype(float)
            D = ST.lognorm(Y)[:, ix].toarray().astype(np.float32)
            for within in (False, True):
                Z, ok = zscore(D, lab[pk] if within else None)
                for s, (e, J) in sets.items():
                    o = pad(Z, ok, e['Pr'][None], e['Al'][None])[0]; n = pad(Z, ok, J['Pr'][:200], J['Al'][:200])[:, 0]
                    acc.setdefault((s, within), []).append([*o, n.mean(), n.std(), (o[0] - n.mean()) / n.std(), (1 + (n >= o[0]).sum()) / 201])
        for (s, within), a in acc.items():
            m = np.median(np.array(a), 0); pr = np.array(a)[:, 0]
            rows.append(dict(tag, score=s, variant='matched', within_region=within, n_nuclei=N_REG * len(regs), n_regions=len(regs), pad=m[0], within_sets=m[1], between_sets=m[2], null_mean=m[3],
                             null_sd=m[4], z=m[5], p=m[6], pad_q05=np.quantile(pr, .05), pad_q95=np.quantile(pr, .95)))
    return rows


def run():
    O = ST.out_dir(); rng = np.random.default_rng(11); X, genes, cells = ST.load(); L = ST.lognorm(X); GI = ST.gene_index(genes); lab = cells.region_label.values; out = []
    for g in ST.GROUPS:
        sel = cells.subclass.isin(ST.GROUPS[g]).values; q = np.flatnonzero(sel & np.isin(lab, ST.REGIONS)); out += one(X, L, q, lab, GI, rng, dict(group=g)); print('adult', g, len(q), flush=True)
    pd.DataFrame(out).to_csv(os.path.join(O, 'pad.csv'), index=False)
    del X, L
    X, genes, cells = SN.load(); X = X.astype(np.float32); L = ST.lognorm(X); GI = ST.gene_index(np.asarray(genes, str)); smp = cells.samplename.values; out = []
    for g in cells.group.unique():
        for a in SN.AGES:
            q = np.flatnonzero((cells.group == g).values & (cells.age == a).values); out += one(X, L, q, smp, GI, rng, dict(group=g, age=a)); print('age', g, a, len(q), flush=True)
    pd.DataFrame(out).to_csv(os.path.join(O, 'pad_ages.csv'), index=False); print('done', flush=True)


if __name__ == '__main__': run()
