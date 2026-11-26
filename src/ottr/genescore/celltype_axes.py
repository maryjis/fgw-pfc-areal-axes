"""Cell types of the Pr-Al atlas (Huang et al. 2026, marmoset neurons, region x cell type) against our areal axes, the way the authors
relate cell types to their Pr-Al index (Pearson of the normalised share with the index, P < 0.05, sign = Pr / Al).
python -m ottr.genescore.celltype_axes <ct.csv> <ct_R2.csv> <ctmeta.csv>
Target = mu of an axis over the adult PFC regions (and, as a control, the Pr-Al index of the same regions). Share: log1p CP10k of the
neuron counts of a region, z over the regions. p = Pearson; q = BH over the 180 types; p_msr = spatial null (Moran, surrogates of the
target over the same regions). Replicate R2 scored with the same target. -> results/tables/gene_axes/celltype_axes.csv, _groups.csv"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from ..config import cfg
from ..registry import axes_model
from . import significance as SG

NS = 5000


def group(n):
    return 'PV-CHC' if n.startswith('PV-CHC') else 'LAMP5-RELN' if n.startswith('LAMP5-RELN') else n.split('-')[0]


def share(m):
    d = np.log1p(m / m.sum(0) * 1e4); return d


def run(f1, f2, fm):
    c = cfg(); A = pd.read_csv(f1, index_col=0).drop('discard', errors='ignore'); B = pd.read_csv(f2, index_col=0).drop('discard', errors='ignore'); idx = pd.read_csv(fm, index_col=0)['Pr-Al-Index']
    Z = axes_model('fgw2d_factors').npz; mu = pd.DataFrame(Z['pb_mu'][:, :5], index=Z['pb_regions'], columns=[f'axis{k}' for k in range(1, 6)])
    low = {r.lower(): r for r in A.columns}; reg = [r for r in mu.index if r.lower() in low]; cols = [low[r.lower()] for r in reg]; M = mu.loc[reg]; M['PrAl_index'] = idx.reindex(cols).values
    print(len(reg), 'regions; missing', [r for r in mu.index if r not in reg], '| not in R2:', [x for x in cols if x not in B.columns], flush=True)
    for k in mu: v = mu[k].sort_values(); print(k, '+', list(v.index[::-1][:4]), '-', list(v.index[:4]))
    C = SG.marmoset_centroids(reg); rng = np.random.default_rng(3); types = [t for t in A.index if t in B.index]; out = []
    SA, SB = share(A[cols]).loc[types].values, share(B[cols]).loc[types].values
    nl = [np.log10(A[cols].sum()).values, np.log10(B[cols].sum()).values]
    for k in M:
        y = M[k].values.astype(float); Sy, _ = SG.surrogates(y, C, NS, rng, 'msr'); Sy = Sy - Sy.mean(0); Sy /= np.linalg.norm(Sy, axis=0)
        for j, t in enumerate(types):
            va, vb = SA[j], SB[j]; r, p = pearsonr(va, y); r2, p2 = pearsonr(vb, y); z = (va - va.mean()) / np.linalg.norm(va - va.mean()); nz = np.abs(z @ Sy)
            out.append(dict(axis=k, type=t, group=group(t), n_regions=len(y), mean_share=100 * float((A[cols].loc[t] / A[cols].sum()).mean()), r=r, p=p, p_msr=(1 + (nz >= abs(r) - 1e-12).sum()) / (NS + 1), r_R2=r2, p_R2=p2, rho=SG.rho(va, y)))
    D = pd.DataFrame(out)
    for k, i in D.groupby('axis').groups.items():
        for q, p in (('q', 'p'), ('q_msr', 'p_msr'), ('q_R2', 'p_R2')): D.loc[i, q] = multipletests(D.loc[i, p], method='fdr_bh')[1]
    s = lambda r, p: np.where(p >= .05, 'ns', np.where(r > 0, '+', '-')); D['call_R1'] = s(D.r, D.p); D['call_R2'] = s(D.r_R2, D.p_R2)
    D['both'] = np.where((D.call_R1 == D.call_R2) & (D.call_R1 != 'ns'), D.call_R1, '')
    D.to_csv(c.tables('gene_axes/celltype_axes.csv'), index=False); D.pivot_table(index='group', columns=['axis', 'call_R1'], values='type', aggfunc='count', fill_value=0).to_csv(c.tables('gene_axes/celltype_axes_groups.csv'))
    return D


if __name__ == '__main__':
    D = run(*sys.argv[1:4])
    print(D.groupby('axis').apply(lambda d: pd.Series(dict(plus=(d.call_R1 == '+').sum(), minus=(d.call_R1 == '-').sum(), q05=(d.q < .05).sum(), msr05=(d.p_msr < .05).sum(), q_msr05=(d.q_msr < .05).sum(), R2plus=(d.call_R2 == '+').sum(), R2minus=(d.call_R2 == '-').sum(),
          both=(d.both != '').sum(), opposite=((d.call_R1 == '+') & (d.call_R2 == '-') | (d.call_R1 == '-') & (d.call_R2 == '+')).sum(), r_R1R2=np.corrcoef(d.r, d.r_R2)[0, 1]))).round(2).to_string())
    for k, d in D.groupby('axis'):
        b = d[(d.both != '')]; print('##', k); g = d.groupby(['group', 'call_R1']).size().unstack(fill_value=0); g2 = d[d.both != ''].groupby(['group', 'both']).size().unstack(fill_value=0); print(g.join(g2.add_suffix(' both')).fillna(0).astype(int).to_string())
        for sg in ('+', '-'): e = b[b.both == sg].sort_values('r', ascending=sg == '-'); print(sg, len(e), '; '.join(f"{t} {r:.2f}/{r2:.2f}{'!' if p < .05 else ''}" for t, r, r2, p in zip(e.type[:14], e.r[:14], e.r_R2[:14], e.p_msr[:14])))
