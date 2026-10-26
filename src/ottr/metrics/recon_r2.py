"""R2 of the reconstructions of the trained models (step 06).

Input per model (ottr.inspire.train --recon_out <prefix> --init_ckpt <final checkpoint>): <prefix>_counts_pred.npy and
_counts_obs.npy (log1p CP10k of the predicted count mean and of the observed counts, float16, all spots in the order of
inputs/<ds>.h5ad) and <prefix>_features.npz (per-gene sums of the feature reconstruction).
Counts: the stored log1p values are turned back into CP10k first (comparing the log of the predicted MEAN with the log of
single noisy counts is biased by Jensen's inequality and gives negative R2):
  spot       : per gene R2 on CP10k over all spots (and the per-gene Pearson r)
  pseudobulk : mean CP10k per section x region x layer (>= 5 spots) for observed and predicted, then log1p, R2 per gene
R2 per gene = 1 - SSE / SST (SST around the gene mean); median over genes and pooled (sum SSE / sum SST).
Features (the auto-encoder targets): log1p CP10k of the genes and its neighbourhood average (axf).
run([(label, prefix)]) -> tables/recon_r2.csv"""
import numpy as np
import pandas as pd
import anndata as ad
from ..config import cfg


def r2(y, p):
    sst = ((y - y.mean(0)) ** 2).sum(0); sse = ((y - p) ** 2).sum(0); keep = sst > 0
    return 1 - sse[keep] / sst[keep], 1 - sse[keep].sum() / sst[keep].sum(), keep


def run(models, ds=None):
    c = cfg(); ds = ds or c.series['name']
    o = ad.read_h5ad(c.inputs(ds), backed='r').obs; lay = o.layer.astype(str).values; reg = o.region.astype(str).values; sec = o.slice.astype(str).values
    ok = np.isin(lay, ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']) & ~np.isin(reg, ['nan', 'None', 'remove', 'unassigned'])
    g = pd.DataFrame({'t': sec, 'r': reg, 'l': lay})[ok].groupby(['t', 'r', 'l']).indices; base = np.flatnonzero(ok); groups = [base[v] for v in g.values() if len(v) >= 5]
    obs = np.load(f'{models[0][1]}_counts_obs.npy', mmap_mode='r'); assert len(obs) == len(o), (len(obs), len(o))
    Y = np.expm1(np.asarray(obs, np.float32)); PBy = np.log1p(np.array([Y[ix].mean(0) for ix in groups])); rows = []
    for lbl, pre in models:
        P = np.expm1(np.asarray(np.load(f'{pre}_counts_pred.npy', mmap_mode='r'), np.float32))
        rg, rp, _ = r2(Y, P); PBp = np.log1p(np.array([P[ix].mean(0) for ix in groups])); qg, qp, _ = r2(PBy, PBp)
        Yc = Y - Y.mean(0); Pc = P - P.mean(0); pr = (Yc * Pc).sum(0) / np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0) + 1e-12); del Yc, Pc
        F = np.load(f'{pre}_features.npz'); fe = {}
        for k in ('genes', 'axf'):
            if f'{k}_n' not in F.files: continue
            n = float(F[f'{k}_n']); sst = F[f'{k}_s2'] - F[f'{k}_s1'] ** 2 / n; kk = sst > 1e-9
            fe[k] = (np.median(1 - F[f'{k}_sse'][kk] / sst[kk]), 1 - F[f'{k}_sse'][kk].sum() / sst[kk].sum())
        rows.append(dict(model=lbl, counts_spot_median_gene_R2=np.median(rg), counts_spot_median_gene_pearson=np.nanmedian(pr), counts_spot_pooled_R2=rp,
                         counts_pseudobulk_median_gene_R2=np.median(qg), counts_pseudobulk_pooled_R2=qp, features_genes_median_R2=fe['genes'][0],
                         features_genes_pooled_R2=fe['genes'][1], features_axf_median_R2=fe.get('axf', (np.nan, np.nan))[0],
                         features_axf_pooled_R2=fe.get('axf', (np.nan, np.nan))[1]))
        print(lbl, 'done', flush=True)
    T = pd.DataFrame(rows); T.to_csv(c.tables('recon_r2.csv'), index=False)
    return T
