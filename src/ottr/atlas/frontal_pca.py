"""PCA of the frontal cortex in the segment atlas (Huang et al. 2026) (step 08).  python -m ottr.atlas.frontal_pca

Data (paths.atlas_segments, exported from Segment_genes_marmoset.rds by scripts/R/segments_export_data.R): slot `data` for
the 5 000 variable genes of their PCA, all 17 525 segments; segments_meta.csv (region, zone VL_region, section, metadata),
segments_pca.csv (their global PCs). Frontal = segments with VL_region in MPFC, OFC, VLPFC, DLPFC, ACC.
Each gene is centred and divided by its SD over the frontal segments (genes / SD, no clipping); PCA (10 components) on the
frontal segments; every segment of the cortex is standardised with the frontal mean / SD and projected on the loadings.
Control: the same PCA on all segments must reproduce their Gene-PC1 (|r| printed).
Per PC 1-6: variance share inside the frontal cortex, eta2 of zone / region / section, Spearman with the numeric metadata,
best global atlas PC, regions and genes at both ends.
-> tables/atlas/frontal_pca_{annotation,scores,loadings,corr}.csv"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from ..config import cfg, ensure

FRONTAL = ['MPFC', 'OFC', 'VLPFC', 'DLPFC', 'ACC']; NPC = 10; NSHOW = 6
# reading of every frontal PC (from the annotation table and the flat maps)
FNAME = {1: 'sections / mitochondrial genes', 2: 'SA (Pr–Al)', 3: 'glia ↔ neurons', 4: 'medial ↔ agranular orbital-insular', 5: 'section-dependent',
         6: 'limbic / polar ↔ cingulate / premotor', 7: 'Gu, A13L, OPro ↔ A8b, OPAl, A14C', 8: 'A24a, OPAl, A13L ↔ A24d, A32V, A25',
         9: 'dorsal ↔ granular orbital (A13 / A14)', 10: 'arch: orbital middle ↔ both SA ends'}


def out_dir():
    return ensure(cfg().tables('atlas'))


def load_atlas():
    O = cfg().path('atlas_segments')
    genes = open(f'{O}/atlas_data_hvg5000_genes.txt').read().split(); segs = open(f'{O}/atlas_data_hvg5000_segments.txt').read().split()
    X = np.fromfile(f'{O}/atlas_data_hvg5000.f32', dtype=np.float32).reshape(len(segs), len(genes)).astype(np.float64)
    S = pd.read_csv(f'{O}/segments_meta.csv', index_col=0).loc[segs]; GP = pd.read_csv(f'{O}/segments_pca.csv', index_col=0).loc[segs]
    return genes, segs, X, S, GP


def eta(v, g):
    s = pd.Series(v); return float(((s.groupby(g).transform('mean') - s.mean()) ** 2).sum() / ((s - s.mean()) ** 2).sum())


def run():
    OUT = out_dir(); genes, segs, X, S, GP = load_atlas()
    fr = S.VL_region.isin(FRONTAL).values; print('frontal segments', fr.sum(), 'of', len(segs), flush=True)
    Zg = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1); g1 = PCA(1, svd_solver='randomized', random_state=0).fit_transform(Zg)[:, 0]
    print('control: |r| global PC1 vs their Gene-PC1 =', round(abs(np.corrcoef(g1, S['Gene-PC1'])[0, 1]), 4), flush=True); del Zg
    mu, sd = X[fr].mean(0), X[fr].std(0); keep = sd > 1e-8; G = np.array(genes)[keep]
    Zf = (X[fr][:, keep] - mu[keep]) / sd[keep]
    pca = PCA(NPC, svd_solver='full', random_state=0).fit(Zf)
    SC = pca.transform((X[:, keep] - mu[keep]) / sd[keep])
    print('genes used', keep.sum(), '| variance shares', np.round(pca.explained_variance_ratio_, 3), flush=True)
    cols = [f'fPC{k + 1}' for k in range(NPC)]
    pd.DataFrame(SC, index=segs, columns=cols).rename_axis('segment').to_csv(f'{OUT}/frontal_pca_scores.csv')
    pd.DataFrame(pca.components_.T, index=G, columns=cols).rename_axis('gene').to_csv(f'{OUT}/frontal_pca_loadings.csv')
    SKIP = {'nCount_RNA', 'nFeature_RNA', 'segment', 'X_axis', 'Y_axis_smooth', 'length', 'umi_real', 'genes_real'}
    NUM = [c for c in S.columns if pd.api.types.is_numeric_dtype(S[c]) and c not in SKIP]
    rows, corr = [], {}
    for k in range(NSHOW):
        v = SC[:, k]; vf = v[fr]
        cf = {f: spearmanr(vf, S[f].values[fr], nan_policy='omit')[0] for f in NUM}; ca = {f: spearmanr(v, S[f], nan_policy='omit')[0] for f in NUM}
        gf = {j: spearmanr(vf, GP[f'PC_{j}'].values[fr])[0] for j in range(1, 11)}; ga = {j: spearmanr(v, GP[f'PC_{j}'])[0] for j in range(1, 11)}
        bf = max(gf, key=lambda j: abs(gf[j])); ba = max(ga, key=lambda j: abs(ga[j]))
        corr[f'fPC{k + 1} frontal'] = cf; corr[f'fPC{k + 1} whole cortex'] = ca
        rm = pd.Series(vf).groupby(S.region.values[fr]).agg(['mean', 'size']); rm = rm[rm['size'] >= 10]['mean'].sort_values()
        l = pd.Series(pca.components_[k], index=G).sort_values(); lo = [g for g in l.index if not g.startswith('LOC')][:8]; hi = [g for g in l.index[::-1] if not g.startswith('LOC')][:8]
        top_f = sorted(cf.items(), key=lambda t: -abs(t[1]))[:4]
        rows.append(dict(pc=k + 1, var_share_frontal=pca.explained_variance_ratio_[k], eta2_zone=eta(vf, S.VL_region.values[fr]), eta2_region=eta(vf, S.region.values[fr]),
                         eta2_section=eta(vf, S.slice.values[fr]), rho_PrAl_frontal=cf['Pr-Al-Index'], rho_PrAl_whole=ca['Pr-Al-Index'], rho_AP_frontal=cf['AP'],
                         best_global_pc_frontal=f'PC{bf}', rho_best_frontal=gf[bf], best_global_pc_whole=f'PC{ba}', rho_best_whole=ga[ba],
                         top_fields_frontal='; '.join(f'{f} {r:+.2f}' for f, r in top_f), regions_low=', '.join(rm.index[:4]), regions_high=', '.join(rm.index[::-1][:4]),
                         genes_low=', '.join(lo), genes_high=', '.join(hi)))
    A = pd.DataFrame(rows); A.to_csv(f'{OUT}/frontal_pca_annotation.csv', index=False); pd.DataFrame(corr).rename_axis('field').to_csv(f'{OUT}/frontal_pca_corr.csv')
    return A


if __name__ == '__main__':
    print(run().round(2).to_string())
