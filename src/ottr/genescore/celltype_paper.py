"""The authors' cell-type analysis (Huang et al. 2026, Fig. 2E-H; notebooks 03, 04; Pr_Al.R) with our axes in place of the Pr-Al index.
python -m ottr.genescore.celltype_paper <ct.csv> <ct_R2.csv> <ctmeta.csv>
Per type: log1p(1e4 x share) over all types of the object (NormalizeData, 'discard' kept), z over regions (no effect on r), Pearson with
the index over the regions, P < 0.05 uncorrected, sign = end. Replicate R2: the same index, cell types of R2. Cell-PC1/2: PCA of the
scaled matrix over all 116 regions (computed anew in R2), correlated with the index.
Index = Pr-Al-Index (116 regions, the authors' analysis) or mu of an axis (30 PFC regions; + = positive mu).
-> results/tables/gene_axes/celltype_paper_{types,summary,pc}.csv, results/pictures/celltype_axes_fig2H.png"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import cfg
from ..registry import axes_model
from .celltype_axes import group

ORDER = ['L2', 'L2/3', 'L3', 'L3/4', 'L3/4/5', 'L4', 'L4/5', 'L5', 'L5/6', 'L6', 'LAMP5', 'LAMP5-RELN', 'PV', 'PV-CHC', 'RELN', 'SST', 'VIP']


def norm(m): return np.log1p(m / m.sum(0) * 1e4)


def cellpc(m):
    X = norm(m).values; sd = X.std(1, ddof=1); X = ((X - X.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1)[:, None]).clip(max=10).T; X = X - X.mean(0); U, s, V = np.linalg.svd(X, full_matrices=False)
    return pd.DataFrame(U[:, :2] * s[:2], index=m.columns, columns=['PC1', 'PC2']), (s ** 2 / (s ** 2).sum())[:2]


def run(f1, f2, fm):
    c = cfg(); A, B = pd.read_csv(f1, index_col=0), pd.read_csv(f2, index_col=0); idx = pd.read_csv(fm, index_col=0)['Pr-Al-Index']; Z = axes_model('fgw2d_factors').npz
    mu = pd.DataFrame(Z['pb_mu'][:, :5], index=Z['pb_regions'], columns=[f'axis{k}' for k in range(1, 6)]); low = {r.lower(): r for r in A.columns}; mu.index = [low[r.lower()] for r in mu.index]
    index = {'PrAl (116 regions)': idx, **{k: mu[k] for k in mu}}; rows, pc = [], []; P = {'R1': cellpc(A), 'R2': cellpc(B)}
    for k, y in index.items():
        for rep, m in (('R1', A), ('R2', B)):
            reg = list(y.index); X = norm(m)[reg]; 
            for t in m.index:
                if t == 'discard' or X.loc[t].std() == 0: continue
                r, p = pearsonr(X.loc[t].values, y.values); rows.append(dict(index=k, rep=rep, type=t, group=group(t), n_regions=len(reg), r=r, p=p, call='ns' if p >= .05 else '+' if r > 0 else '-', cells=int(m.loc[t, reg].sum())))
            E, ev = P[rep]
            for j in ('PC1', 'PC2'): r, p = pearsonr(E.loc[reg, j], y); pc.append(dict(index=k, rep=rep, pc=j, explained=ev[int(j[-1]) - 1], n_regions=len(reg), r=r, p=p))
    D = pd.DataFrame(rows); D.to_csv(c.tables('gene_axes/celltype_paper_types.csv'), index=False); PC = pd.DataFrame(pc); PC.to_csv(c.tables('gene_axes/celltype_paper_pc.csv'), index=False)
    s = D.groupby(['index', 'rep', 'group', 'call']).agg(types=('type', 'size'), cells=('cells', 'sum')).reset_index(); s.to_csv(c.tables('gene_axes/celltype_paper_summary.csv'), index=False)
    return D, PC, s


def plot(s):
    INK, INK2 = '#14161a', '#5b5e66'; COL = {'+': '#c9480f', '-': '#1c5fb0'}; ks = list(dict.fromkeys(['PrAl (116 regions)'] + [f'axis{i}' for i in range(1, 6)]))
    ends = {'PrAl (116 regions)': ('Pr', 'Al'), 'axis1': ('A8C end', 'A25 end'), 'axis2': ('A25 end', 'ProM end'), 'axis3': ('A24b end', 'OPAl end'), 'axis4': ('A24c end', 'A10 end'), 'axis5': ('A13L end', 'A14R end')}
    q = s[s.rep == 'R1']; fig, ax = plt.subplots(2, 6, figsize=(23, 8.6), sharex=True, facecolor='white'); x = np.arange(len(ORDER))
    for j, k in enumerate(ks):
        for i, v in enumerate(('types', 'cells')):
            a = ax[i, j]; sc = 1 if v == 'types' else 1e-5
            for sg, d in (('+', 1), ('-', -1)):
                h = q[(q['index'] == k) & (q.call == sg)].set_index('group')[v].reindex(ORDER).fillna(0) * sc; a.bar(x, d * h, color=COL[sg], width=.78)
            a.axhline(0, color=INK2, lw=.8); a.set_xticks(x); a.set_xticklabels(ORDER, rotation=90, fontsize=8, color=INK)
            for sp in ('top', 'right'): a.spines[sp].set_visible(False)
            for sp in ('left', 'bottom'): a.spines[sp].set_color('#d5d4cf')
            a.tick_params(length=0, labelsize=8, colors=INK2); a.yaxis.set_major_formatter(lambda t, _: f'{abs(t):g}')
            if j == 0: a.set_ylabel('Number of cell types' if i == 0 else 'Cells (x 10^5)', color=INK, fontsize=9.5)
        n = {sg: int(q[(q['index'] == k) & (q.call == sg)].types.sum()) for sg in '+-'}
        ax[0, j].set_title(f'{"Pr-Al index (116 regions)" if j == 0 else "Axis " + k[-1] + " (30 PFC regions)"}\n{n["+"]} up: {ends[k][0]}, {n["-"]} down: {ends[k][1]}', fontsize=10, fontweight='bold', color=INK, loc='left')
    m = max(max(abs(v) for v in a.get_ylim()) for a in ax[0])
    for a in ax[0]: a.set_ylim(-m, m)
    for a in ax[1]: v = max(abs(t) for t in a.get_ylim()); a.set_ylim(-v, v)
    fig.suptitle('Cell types associated with an index, as Fig. 2H of Huang et al. 2026 (Pearson, P < 0.05 uncorrected, animal I)', x=.01, ha='left', fontsize=13, fontweight='bold', color=INK)
    fig.text(.01, .005, 'Up (orange): positive r with the index; down (blue): negative r. About 9 of 180 types are expected at P < 0.05 by chance. Bottom row: own scale in every panel, cells counted in the regions used. First column: the authors\' own index on the whole cortex.', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, .03, 1, .95)); fig.savefig(cfg().pictures('celltype_axes_fig2H.png'), dpi=140)


if __name__ == '__main__':
    D, PC, s = run(*sys.argv[1:4]); plot(s); w = D.pivot_table(index=['index', 'type'], columns='rep', values=['r', 'call'], aggfunc='first')
    print(D.groupby(['index', 'rep']).call.value_counts().unstack(fill_value=0).to_string()); print(PC.round(3).to_string())
    for k in D['index'].unique():
        a = D[(D['index'] == k) & (D.rep == 'R1')].set_index('type'); b = D[(D['index'] == k) & (D.rep == 'R2')].set_index('type').reindex(a.index); print(k, 'r(R1,R2)', round(np.corrcoef(a.r, b.r)[0, 1], 2), 'same', ((a.call == b.call) & (a.call != 'ns')).sum(), 'opposite', (((a.call == '+') & (b.call == '-')) | ((a.call == '-') & (b.call == '+'))).sum())
    for k in ('axis3', 'axis4', 'axis2', 'axis5'):
        a = D[(D['index'] == k) & (D.rep == 'R1') & (D.call != 'ns')].sort_values('r'); print(k, '+', '; '.join(f'{t} {r:.2f}' for t, r in zip(a.type[::-1][:8], a.r[::-1][:8]) if r > 0), '| -', '; '.join(f'{t} {r:.2f}' for t, r in zip(a.type[:8], a.r[:8]) if r < 0))
