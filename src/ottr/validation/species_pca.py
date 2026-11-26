"""E3, E4 and the cell-type composition rows of the validation plan.   python -m ottr.validation.species_pca
E3 components of other species: PCA of the region x gene matrices of Huang et al. 2026 (macaque Stereo-seq pseudobulks: log1p CP10k;
   human microarray values as given), genes z-scored, (a) whole cortex, (b) the frontal regions of data/species_region_map.tsv only.
   Frontal region scores of PC1-3 vs our mu carried through the region map. Primary: mu axis k x best of PC1-3 of the frontal PCA (10 tests).
E4 macaque, composition vs expression: partial Spearman of the macaque gene score with mu given Cell-PC1..3 of the macaque neuron
   composition (Region_celltype_macaque; replicates R2, R3 with their own PCs). Primary: axes 1, 3 with R1.
Cell-type composition (marmoset, Huang et al. replicates R1 / R2): Cell-PC1..3 of the 116-region composition at our 30 regions vs mu;
   primary: mu axis k x best of Cell-PC1..3 (R1), replication = R2 same component sign.
-> results/tables/validation/rows_species_pca.csv, rows_composition.csv"""
import os
import numpy as np
import pandas as pd
from ..config import cfg, ROOT
from ..genescore import species as SP
from . import common as C

COMP = os.path.join(cfg().work, 'validation', 'composition')


def pcs(M, k=3):
    M = np.asarray(M, float); Z = (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1); U, s, _ = np.linalg.svd(Z, full_matrices=False); return U[:, :k] * s[:k], s[:k] ** 2 / (s ** 2).sum()


def region_map(sp):
    M = pd.read_csv(os.path.join(ROOT, 'data', 'species_region_map.tsv'), sep='\t', dtype=str); M = M[M.species == sp]; return dict(zip(M.their_region, M.our_regions.str.split(',')))


def expr(sp):
    X, genes, R, kind = SP.load(sp); X = X.toarray().astype(float)
    if kind == 'int32': X = np.log1p(X / X.sum(1, keepdims=True) * 1e4)
    keep = X.std(0) > 0; return pd.DataFrame(X[:, keep], index=R.region.astype(str).values)


def e3(rng):
    rows = []
    for sp in ('macaque', 'human'):
        E = expr(sp); mp = region_map(sp); fr = [r for r in E.index if r in mp]; mapping = [mp[r] for r in fr]
        for scope, rws in (('frontal', fr), ('whole cortex', list(E.index))):
            S, ev = pcs(E.loc[rws].values); S = pd.DataFrame(S, index=rws, columns=['PC1', 'PC2', 'PC3']).loc[fr]
            for k in range(1, 6):
                r = C.test_best(f'mu axis{k}', S.values, list(S.columns), mapping=mapping, rng=rng, domain='species components (E3)', dataset=f'{sp} {scope} PCA', species=sp, family='primary' if scope == 'frontal' else 'exploratory')
                r.update(feature=f'best of PC1-3 ({scope} PCA): {r.pop("best")}', note=f'variance {", ".join(f"{v:.0%}" for v in ev)}; {r.pop("all_rho")}'); rows.append(r)
    return C.save(rows, 'species_pca')


def composition_pcs(name):
    M = pd.read_csv(os.path.join(COMP, f'{name}.csv'), index_col=0); M = M.drop('discard', errors='ignore'); L = np.log1p(M / M.sum(0) * 1e4).T
    S, ev = pcs(L.values); return pd.DataFrame(S, index=L.index, columns=['Cell-PC1', 'Cell-PC2', 'Cell-PC3']), ev


def e4(rng):
    rows = []; sc = pd.read_csv(cfg().tables('gene_axes/species/scores_macaque.csv'), dtype={'region': str}).set_index('region'); mp = region_map('macaque'); fr = [r for r in sc.index if r in mp]; mapping = [mp[r] for r in fr]
    for rep in ('', '_R2', '_R3'):
        P, ev = composition_pcs(f'Region_celltype_macaque{rep}'); Z = P.reindex(fr).values
        for s, k in (('axis1', 1), ('axis3', 3), ('axis2', 2), ('axis4', 4), ('axis5', 5), ('PrAl', 1)):
            r = C.test(f'mu axis{k}', sc.loc[fr, s].values, mapping=mapping, covariates=Z, rng=rng, domain='macaque composition vs expression (E4)', dataset=f'macaque gene score | Cell-PC1-3 of R{rep[-1] if rep else 1}',
                       species='macaque', feature=f'{s} gene score, partial | Cell-PC1..3', family='primary' if (rep == '' and s in ('axis1', 'axis3')) else 'exploratory', note=f'Cell-PC variance {", ".join(f"{v:.0%}" for v in ev)}')
            if r: rows.append(r)
    D = pd.DataFrame(rows)
    for s in D.feature.unique():                                                                       # replication: R2 and R3 with the same sign as R1
        q = D.feature == s; ref = np.sign(D.loc[q & D.dataset.str.endswith('R1'), 'spearman'].values[0]); reps = np.sign(D.loc[q & ~D.dataset.str.endswith('R1'), 'spearman'].values)
        D.loc[q & D.dataset.str.endswith('R1'), 'replication'] = 'same' if (reps == ref).all() else 'opposite'
    return C.save(D.to_dict('records'), 'e4_macaque')


def composition(rng):
    O = C.ours(); low = lambda idx: {r.lower(): r for r in idx}; rows = []
    P1, ev1 = composition_pcs('Region_celltype_marmoset'); P2, ev2 = composition_pcs('Region_celltype_marmoset_R2')
    a1, a2 = low(P1.index), low(P2.index); X1 = P1.loc[[a1[r.lower()] for r in O.index]]; X2 = P2.loc[[a2[r.lower()] for r in O.index]]
    for k in range(1, 6):
        r = C.test_best(f'mu axis{k}', X1.values, list(X1.columns), rng=rng, domain='cell-type composition', dataset='Huang et al. 2026 neuron composition, animal I (R1)', species='marmoset', family='primary')
        b = r.pop('best'); j = list(X1.columns).index(b); r2 = np.corrcoef(np.argsort(np.argsort(O[f'mu axis{k}'].values)), np.argsort(np.argsort(X2[b].values)))[0, 1]
        sgn = np.sign(np.corrcoef(X1[b], X2[b])[0, 1])                                                  # PC signs are arbitrary: align R2 to R1 over the 30 regions
        r.update(feature=f'best of Cell-PC1-3: {b}', replication='same' if np.sign(r2 * sgn) == np.sign(r['spearman']) else 'opposite', note=f'R2 {b}: rho {r2 * sgn:+.2f} (sign aligned); R1 variance {", ".join(f"{v:.0%}" for v in ev1)}; {r.pop("all_rho")}'); rows.append(r)
    T = pd.read_csv(cfg().tables('gene_axes/celltype_paper_types.csv'))
    for k in range(1, 6):
        a = T[(T['index'] == f'axis{k}') & (T.rep == 'R1')].set_index('type'); b = T[(T['index'] == f'axis{k}') & (T.rep == 'R2')].set_index('type').reindex(a.index)
        same = int(((a.call == b.call) & (a.call != 'ns')).sum()); opp = int((((a.call == '+') & (b.call == '-')) | ((a.call == '-') & (b.call == '+'))).sum())
        rows.append(dict(domain='cell-type composition', dataset='Huang et al. method (Pearson per cell type, P < 0.05)', species='marmoset', axis=str(k), ours=f'mu axis{k}', feature='cell types associated (R1 / R2)', n=30, family='descriptive',
                         spearman=float(np.corrcoef(a.r, b.r)[0, 1]), note=f'R1: {int((a.call == "+").sum())} up / {int((a.call == "-").sum())} down of 180; both replicates same sign: {same}; opposite: {opp}; spearman column = r(R1, R2) over types'))
    return C.save(rows, 'composition')


if __name__ == '__main__':
    rng = np.random.default_rng(29); pd.set_option('display.width', 250); cols = ['domain', 'dataset', 'ours', 'feature', 'n', 'spearman', 'p_perm', 'q_perm', 'p_msr', 'q_msr', 'p_vario', 'q_vario', 'replication', 'verdict']
    for f in (e3, e4, composition): D = f(rng); print(D[cols].round(4).to_string(index=False)); print(D[['note']].to_string(index=False)[:3000])
