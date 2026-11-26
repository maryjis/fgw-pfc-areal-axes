"""E3b (exploratory, added after the main run): PCA of the region x gene matrices of other species AFTER regressing out the
cell-type composition, so that the components are closer to our areal model (which removes the layer profile).
python -m ottr.validation.residual_pca

macaque  Stereo-seq region pseudobulks of Huang et al. 2026 (Chen et al. 2023), log1p CP10k, 143 regions; covariates = Cell-PC1..k of
         the macaque neuron composition (Region_celltype_macaque, replicate R1, same regions); k = 3 (as E4) and 10 (sensitivity).
human    Siletti et al. 2023 snRNA-seq: pseudobulk of all nuclei per cortical dissection (raw counts summed, log1p CP10k); covariates =
         Cell-PC1..k of the human composition of the same dissections (Region_celltype_human of Huang et al.). The human microarray
         parcels (E3) have no composition, so they cannot be corrected.
Per species and scope (whole cortex; frontal regions of the region map): every gene regressed on [1, Cell-PC1..k] by OLS over the
regions of the scope, residuals z-scored per gene, SVD -> PC1-5. k = 0 = no correction (E3 repeated on the same matrices).
Test: mu axis j x best of PC1-3 at the frontal regions (ottr.validation.common.test_best: every null takes the max over PCs);
all PC1-5 correlations in the note. Family 'exploratory' (not fixed before the run).
NOTE: removing the composition is NOT what our areal model does (it removes the layer profile and the section sequencing depth only;
the change of cell types between regions is part of the areal signal). The analogues of our model are model_like():
macaque  only region-level expression exists: every gene regressed on the layer shares of the region (cells of L1..L5 / all cells,
         from Region_L<l>_celltype_macaque; a missing layer = 0, e.g. no L4 in agranular areas) and log(UMI per cell) (depth).
human    Siletti blocks dissection x supercluster (>= 50 nuclei) = the region x layer blocks of our model: log1p CP10k of the summed
         counts, the supercluster profile removed (centred within each supercluster), log UMI per nucleus of the block regressed
         out, the superclusters of a dissection concatenated into one row (superclusters present in >= 90 % of dissections,
         missing blocks = 0 after centring), each column scaled to SD 1, SVD -> PC1-5 (whole cortex and frontal).
-> work/validation/siletti_pseudobulk.parquet, siletti_pb_super.parquet, results/tables/validation/rows_residual_pca.csv,
   rows_model_like_pca.csv, residual_pca_variance.csv, model_like_pca_variance.csv"""
import os, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from ..config import cfg, ensure
from . import common as C, species_pca as SP, siletti as SL

KS = (0, 3, 10)


def siletti_pseudobulk():
    p = os.path.join(ensure(os.path.join(cfg().work, 'validation')), 'siletti_pseudobulk.parquet')
    if os.path.exists(p): return pd.read_parquet(p)
    with Pool(6) as P: rows = P.map(_one_pb, sorted(glob.glob(f'{SL.DIR}/*.h5ad')))
    D = pd.concat(rows, axis=1).T.fillna(0); D.to_parquet(p); return D


def _one_pb(f):
    import anndata as ad
    a = ad.read_h5ad(f); d = a.obs.dissection.astype(str).map(SL.short).unique(); assert len(d) == 1, d
    s = np.asarray(a.X.sum(0)).ravel(); g = a.var.feature_name.astype(str).str.upper().values
    return pd.Series(s, index=g).groupby(level=0).sum().rename(d[0])


def human():
    D = siletti_pseudobulk(); X = np.log1p(D.values / D.values.sum(1, keepdims=True) * 1e4); E = pd.DataFrame(X, index=D.index).loc[:, X.std(0) > 0]
    Cm = pd.read_csv(os.path.join(SP.COMP, 'Region_celltype_human.csv'), index_col=0); Cm.columns = [c.replace('_', '-') for c in Cm.columns]
    E.index = [r.replace('_', '-').replace('A44-A45', 'A44-A45') for r in E.index]
    alias = {'FI': 'Fl'}; cols = {r: alias.get(r, r) for r in E.index}; Cm = Cm.rename(columns={v: k for k, v in cols.items()})
    L = np.log1p(Cm / Cm.sum(0) * 1e4).T; common = [r for r in E.index if r in L.index]
    print('human dissections: pseudobulk', len(E), 'with composition', len(common), 'missing', sorted(set(E.index) - set(common)), flush=True)
    mp = {('A44-A45' if k == 'A44-A45' else k): v for k, v in SL.MAP.items()}
    return E.loc[common], L.loc[common], mp


def macaque():
    E = SP.expr('macaque'); M = pd.read_csv(os.path.join(SP.COMP, 'Region_celltype_macaque.csv'), index_col=0).drop('discard', errors='ignore')
    L = np.log1p(M / M.sum(0) * 1e4).T.reindex(E.index); return E, L, SP.region_map('macaque')


def residual_pcs(E, L, k, n=5):
    Z = (E.values - E.values.mean(0)) / np.where(E.values.std(0) > 0, E.values.std(0), 1)
    if k:
        Lz = L.values - L.values.mean(0); U, s, _ = np.linalg.svd(Lz / np.where(L.values.std(0) > 0, L.values.std(0), 1), full_matrices=False)
        A = np.column_stack([np.ones(len(Z)), U[:, :k]]); Z = Z - A @ np.linalg.lstsq(A, Z, rcond=None)[0]
        Z = Z / np.where(Z.std(0) > 1e-9, Z.std(0), 1)
    U, s, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False); return pd.DataFrame(U[:, :n] * s[:n], index=E.index, columns=[f'PC{i}' for i in range(1, n + 1)]), s[:n] ** 2 / (s ** 2).sum()


def run(rng):
    rows, var = [], []
    for sp, (E, L, mp) in (('macaque', macaque()), ('human', human())):
        fr = [r for r in E.index if r in mp]; mapping = [mp[r] for r in fr]
        for scope, rws in (('whole cortex', list(E.index)), ('frontal', fr)):
            for k in KS:
                if k and len(rws) - k < 8: continue                       # too few regions left after the regression
                S, ev = residual_pcs(E.loc[rws], L.loc[rws], k); var.append(dict(species=sp, scope=scope, k=k, n_regions=len(rws), **{f'PC{i + 1}': v for i, v in enumerate(ev)}))
                Sf = S.loc[fr]
                for j in range(1, 6):
                    r = C.test_best(f'mu axis{j}', Sf[['PC1', 'PC2', 'PC3']].values, ['PC1', 'PC2', 'PC3'], mapping=mapping, rng=rng, domain='species residual components (E3b)',
                                    dataset=f'{sp} {scope} PCA, composition removed (Cell-PC1..{k})' if k else f'{sp} {scope} PCA, no correction', species=sp, family='exploratory')
                    al = '; '.join(f'{c} {pd.Series(C.carry(C.ours()[f"mu axis{j}"], mapping)[:, 0]).rank().corr(pd.Series(Sf[c].values).rank()):+.2f}' for c in S.columns)
                    r.update(feature=f'best of PC1-3: {r.pop("best")}', note=f'n regions in PCA {len(rws)}; variance PC1-5 {", ".join(f"{v:.0%}" for v in ev)}; PC1-5 rho: {al}'); r.pop('all_rho'); rows.append(r)
    pd.DataFrame(var).to_csv(os.path.join(C.out_dir(), 'residual_pca_variance.csv'), index=False)
    return C.save(rows, 'residual_pca')


def _one_pb_super(f):
    import anndata as ad
    a = ad.read_h5ad(f); d = a.obs.dissection.astype(str).map(SL.short).unique()[0]; g = a.var.feature_name.astype(str).str.upper().values
    _, first = np.unique(g, return_index=True); keep = np.sort(first); out = []
    for sc, ix in a.obs.groupby(a.obs.supercluster_term.astype(str), observed=True).indices.items():
        s = np.asarray(a.X[ix][:, keep].sum(0)).ravel(); out.append(pd.Series(s, index=g[keep]).rename((d, sc, len(ix), float(a.obs.total_UMIs.values[ix].mean()))))
    return out


def siletti_pb_super():
    p = os.path.join(ensure(os.path.join(cfg().work, 'validation')), 'siletti_pb_super.parquet')
    if os.path.exists(p): return pd.read_parquet(p)
    with Pool(6) as P: rows = sum(P.map(_one_pb_super, sorted(glob.glob(f'{SL.DIR}/*.h5ad'))), [])
    D = pd.concat(rows, axis=1).T.fillna(0); D.index = pd.MultiIndex.from_tuples(D.index, names=['dissection', 'supercluster', 'n', 'umi']); D.to_parquet(p); return D


def svd_scores(Z, n=5):
    Z = Z - Z.mean(0); Z = Z / np.where(Z.std(0) > 1e-9, Z.std(0), 1); U, s, _ = np.linalg.svd(Z, full_matrices=False); return U[:, :n] * s[:n], s[:n] ** 2 / (s ** 2).sum()


def macaque_layer_depth(E):
    C = pd.DataFrame({l: pd.read_csv(os.path.join(SP.COMP, f'Region_L{l}_celltype_macaque.csv'), index_col=0).drop('discard', errors='ignore').sum(0) for l in range(1, 7)}).reindex(E.index).fillna(0)
    R = pd.read_csv(os.path.join(cfg().work, 'species', 'macaque', 'regions.tsv'), sep='\t', dtype={'region': str}).set_index('region')
    sh = C.div(C.sum(1), axis=0); dep = np.log(R.loc[E.index, 'nCount_RNA'].values / C.sum(1).values); return np.column_stack([sh[[1, 2, 3, 4, 5]].values, dep])


def human_blocks(rows_keep=None, nmin=50, share=0.9, ngenes=3000):
    D = siletti_pb_super(); idx = D.index.to_frame(index=False); ok = (idx.n >= nmin).values; D, idx = D[ok], idx[ok].reset_index(drop=True)
    X = np.log1p(D.values / D.values.sum(1, keepdims=True) * 1e4); dis = sorted(idx.dissection.unique()) if rows_keep is None else rows_keep
    cnt = idx[idx.dissection.isin(dis)].groupby('supercluster').dissection.nunique(); scs = sorted(cnt[cnt >= share * len(dis)].index)
    m = idx.dissection.isin(dis).values & idx.supercluster.isin(scs).values; X, idx = X[m], idx[m].reset_index(drop=True)
    hv = np.argsort(-X.var(0))[:ngenes]; X = X[:, hv]
    for sc in scs: q = (idx.supercluster == sc).values; X[q] -= X[q].mean(0)                       # supercluster profile (f_l)
    c = np.log(idx.umi.values); c = c - c.mean(); X -= np.outer(c, (c @ X) / (c @ c))                 # depth of the block (theta c)
    M = np.zeros((len(dis), len(scs) * X.shape[1])); pos = {d: i for i, d in enumerate(dis)}
    for j, sc in enumerate(scs):
        q = (idx.supercluster == sc).values
        for x, d in zip(X[q], idx.dissection[q]): M[pos[d], j * X.shape[1]:(j + 1) * X.shape[1]] = x
    return pd.DataFrame(M, index=dis), scs


def model_like(rng):
    rows, var = [], []
    E, _, mp = macaque(); Zc = macaque_layer_depth(E); fr = [r for r in E.index if r in mp]; mapping = [mp[r] for r in fr]
    for scope, rws in (('whole cortex', list(E.index)), ('frontal', fr)):
        ii = [list(E.index).index(r) for r in rws]; Z = E.loc[rws].values; Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
        A = np.column_stack([np.ones(len(ii)), Zc[ii]]); Z = Z - A @ np.linalg.lstsq(A, Z, rcond=None)[0]; S, ev = svd_scores(Z)
        rows += _tests(pd.DataFrame(S, index=rws, columns=[f'PC{i}' for i in range(1, 6)]), ev, fr, mapping, rng, 'macaque', scope, 'layer shares L1-L5 + depth removed', len(rws)); var.append(dict(species='macaque', scope=scope, n_regions=len(rws), **{f'PC{i + 1}': v for i, v in enumerate(ev)}))
    mph = SL.MAP; dis_all = sorted(siletti_pb_super().index.get_level_values('dissection').unique()); frh = [d for d in dis_all if d in mph]
    for scope, dl in (('whole cortex', dis_all), ('frontal', frh)):
        M, scs = human_blocks(dl); S, ev = svd_scores(M.values)
        rows += _tests(pd.DataFrame(S, index=M.index, columns=[f'PC{i}' for i in range(1, 6)]), ev, frh, [mph[d] for d in frh], rng, 'human', scope, f'supercluster profile + depth removed, {len(scs)} superclusters', len(M))
        var.append(dict(species='human', scope=scope, n_regions=len(M), **{f'PC{i + 1}': v for i, v in enumerate(ev)}))
    pd.DataFrame(var).to_csv(os.path.join(C.out_dir(), 'model_like_pca_variance.csv'), index=False); return C.save(rows, 'model_like_pca')


def _tests(S, ev, fr, mapping, rng, sp, scope, how, nreg):
    Sf = S.loc[fr]; out = []
    for j in range(1, 6):
        r = C.test_best(f'mu axis{j}', Sf.values, list(S.columns), mapping=mapping, rng=rng, domain='species model-like components (E3c)', dataset=f'{sp} {scope} PCA, {how}', species=sp, family='exploratory')
        x = pd.Series(C.carry(C.ours()[f'mu axis{j}'], mapping)[:, 0]); al = '; '.join(f'{c} {x.rank().corr(pd.Series(Sf[c].values).rank()):+.2f}' for c in S.columns)
        r.update(feature=f'best of PC1-5: {r.pop("best")}', note=f'n units in PCA {nreg}; variance PC1-5 {", ".join(f"{v:.0%}" for v in ev)}; PC1-5 rho: {al}'); r.pop('all_rho'); out.append(r)
    return out


if __name__ == '__main__' and os.environ.get('MODEL_LIKE'):
    pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 120); R = model_like(np.random.default_rng(37))
    print(pd.read_csv(os.path.join(C.out_dir(), 'model_like_pca_variance.csv')).round(3).to_string(index=False))
    print(R[['dataset', 'ours', 'feature', 'n', 'spearman', 'p_perm', 'p_msr', 'p_vario', 'verdict', 'note']].round(4).to_string(index=False))
elif __name__ == '__main__':
    pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 120); R = run(np.random.default_rng(31))
    print(pd.read_csv(os.path.join(C.out_dir(), 'residual_pca_variance.csv')).round(3).to_string(index=False))
    print(R[['dataset', 'ours', 'feature', 'n', 'spearman', 'p_perm', 'p_msr', 'p_vario', 'q_msr', 'q_vario', 'verdict', 'note']].round(4).to_string(index=False))
