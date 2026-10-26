"""Correspondence of our axes with the frontal PCA of the atlas, per model (step 08).

Our axes are the segment means of our spots on the atlas flat map (ottr.atlas.flatmap.project): areal axes evaluated on spots,
column PCA, spot PCA, spot-level model, components 1-5. Spearman with fPC1-10 over the segments of T408-T428; best |rho| per axis.
match(models)  models: [(model label, flatmap output name, axes label inside that run)]
-> tables/atlas/axes_vs_frontal_models.csv"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ..config import cfg

KINDS = [('areal', 'pseudobulk areal'), ('column', 'column PCA'), ('spot', 'spot PCA'), ('spotmodel', 'spot-level model')]


def match(models):
    c = cfg(); F = pd.read_csv(c.tables('atlas/frontal_pca_scores.csv'), index_col=0); rows = []
    for m, fm, lab in models:
        V = pd.read_csv(c.tables(f'atlas/{fm}/segment_values.csv'), index_col=0); R = pd.read_csv(c.tables(f'atlas/{fm}/segment_matches.csv'))
        R = R[R.model == lab]; Ff = F.loc[V.index]
        for kind, kl in KINDS:
            for k in range(1, 6):
                v = V[f'{lab}|{kind}|{k}']; rho = {j: spearmanr(v, Ff[f'fPC{j}'], nan_policy='omit')[0] for j in range(1, 11)}; jb = max(rho, key=lambda j: abs(rho[j]))
                r = R[(R.kind == kind) & (R.comp == k)].iloc[0]
                rows.append(dict(model=m, analysis=kl, comp=k, reading=r.reading, share=r.var_share, best_fPC=jb, rho_best=rho[jb], **{f'rho_fPC{j}': rho[j] for j in range(1, 11)}))
    S = pd.DataFrame(rows); S.to_csv(c.tables('atlas/axes_vs_frontal_models.csv'), index=False)
    return S
