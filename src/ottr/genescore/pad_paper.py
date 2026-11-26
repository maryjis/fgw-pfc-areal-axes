"""PAD exactly as described by Huang et al. 2026 (Materials and methods, 'Calculation of PAD and gene coexpression matrix'): snRNA-seq,
Pearson gene x gene correlation over the cells of a subclass, Pr500 / Al500; PAD = mean of the Pr-Pr and Al-Al blocks - mean of Pr-Al.
python -m ottr.genescore.pad_paper
Not stated in the paper and chosen here: log1p CP10k; all cortical nuclei of the subclass and age pooled; genes without variance dropped.
pad = diagonal included (plain mean of the matrix); pad_nodiag = without the 1s of the diagonal (the gene with itself).
-> results/tables/gene_axes/snrna_types/pad_paper_axes.csv (PrAl and axes 1-5; pad_paper.csv = the first run, PrAl only), results/pictures/snrna_pad_heatmap.png (--plot)"""
import os
import numpy as np
import pandas as pd
from ..config import cfg
from . import snrna as SN, snrna_types as ST

GROUPS = dict(ST.GROUPS); GROUPS.update({'L6 IT Car3': ['L6 IT Car3'], 'L5/6 NP': ['L5/6 NP'], 'L6b': ['L6b'], 'Chandelier': ['Chandelier'], 'Sncg': ['Sncg'], 'Micro-PVM': ['Micro-PVM'], 'Endo': ['Endo'], 'VLMC': ['VLMC']})
DIRS = {a: f'snrna_types_{a}' for a in ('P0', 'P32', '3M', '7M')}; DIRS['adult'] = 'snrna_types'


def blocks(D, np_):
    sd = D.std(0); ok = sd > 1e-8; Z = (D[:, ok] - D[:, ok].mean(0)) / sd[ok]; n = len(Z); k = int(ok[:np_].sum()); m = int(ok[np_:].sum())
    a, b = Z[:, :k].sum(1, dtype=np.float64), Z[:, k:].sum(1, dtype=np.float64); I, IV, II = (a @ a) / n, (b @ b) / n, (a @ b) / n
    return dict(n_pr=k, n_al=m, prpr=I / k ** 2, alal=IV / m ** 2, pral=II / (k * m), prpr_nd=(I - k) / (k * (k - 1)), alal_nd=(IV - m) / (m * (m - 1)))


def run():
    out = []
    for age, d in DIRS.items():
        p = os.path.join(cfg().work, d); X, genes, cells = SN.load(p); cells = cells.reset_index(drop=True)
        f = pd.read_csv('/workspace/preprocess_ctx/data/8f_metarefine_all_metadata.csv', low_memory=False).set_index('seurat_cell_id').loc[cells.cell]
        assert (f.subclass.values == cells.subclass.values).all() and (f.age.values == age).all(); cells['class_'] = f['class'].values
        L = ST.lognorm(X); del X; GI = ST.gene_index(np.asarray(genes, str)); L = L.tocsc(); M = {s: L[:, np.r_[v['Pr'], v['Al']]].tocsr() for s, v in GI.items()}; del L
        sel = {g: cells.subclass.isin(v).values for g, v in GROUPS.items()}; sel.update({f'CLASS {c}': (cells.class_ == c).values for c in ('ExN', 'InN', 'NNC')})
        for g, w in sel.items():
            q = np.flatnonzero(w)
            if len(q) < 100: continue
            for s, A in M.items():
                r = blocks(A[q].toarray().astype(np.float32), len(GI[s]['Pr'])); out.append(dict(age=age, group=g, score=s, n_nuclei=len(q), **r, pad=(r['prpr'] + r['alal']) / 2 - r['pral'], pad_nodiag=(r['prpr_nd'] + r['alal_nd']) / 2 - r['pral']))
        print(age, len(cells), flush=True)
    pd.DataFrame(out).to_csv(os.path.join(ST.out_dir(), 'pad_paper_axes.csv'), index=False); print('done', flush=True)


def plot(scores=('PrAl', 'axis1'), name='snrna_pad_heatmap.png'):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    P = pd.read_csv(os.path.join(ST.out_dir(), 'pad_paper_axes.csv')).rename(columns={'pad_nodiag': 'v'}); ages = list(DIRS); INK, INK2 = '#14161a', '#5b5e66'
    blocks_ = [('Excitatory', ['L2/3 IT + ExN_dev', 'L2/3 IT (no ExN_dev)', 'L4 IT', 'L5 IT', 'L6 IT', 'L6 IT Car3', 'L5 ET', 'L6 CT', 'L5/6 NP', 'L6b']), ('Inhibitory', ['Pvalb', 'Chandelier', 'Sst', 'Vip', 'Lamp5', 'Sncg']),
               ('Non-neuronal', ['Astro', 'Oligo', 'OPC', 'Micro-PVM', 'Endo', 'VLMC'])]
    rows = []; ys = {}; y = 0
    for b, gs in blocks_:
        for g in gs: ys[g] = y; y += 1
        y += 0.5
    cm = LinearSegmentedColormap.from_list('seq', ['#eef4fc', '#9cc3ee', '#2a78d6', '#0f3f7a'])
    neg = LinearSegmentedColormap.from_list('neg', ['#eef4fc', '#f6b48f', '#eb6834'])
    fig, ax = plt.subplots(1, len(scores), figsize=(5.6 * len(scores) + 1.6, 9.4), facecolor='white', sharey=True)
    for a, s in zip(ax, scores):
        T = P[P.score == s].pivot(index='group', columns='age', values='v')[ages] * 1000; N = P[P.score == s].pivot(index='group', columns='age', values='n_nuclei')[ages]; vmax = float(np.nanmax(np.abs(T.loc[list(ys)].values)))
        for g, yy in ys.items():
            for j, age in enumerate(ages):
                v = T.loc[g, age]; f = max(v, 0) / vmax
                a.add_patch(plt.Rectangle((j + 0.02, yy + 0.02), 0.96, 0.96, color=cm(f) if v >= 0 else neg(min(-v / vmax, 1)))); low = N.loc[g, age] < 500
                a.text(j + .5, yy + .5, f'{v:.1f}' + ('*' if low else ''), ha='center', va='center', fontsize=9, color='white' if f > .5 else INK)
        a.set_xlim(0, len(ages)); a.set_ylim(y, -0.1); a.set_xticks(np.arange(len(ages)) + .5); a.set_xticklabels(ages, color=INK); a.xaxis.tick_top()
        a.set_yticks([yy + .5 for yy in ys.values()]); a.set_yticklabels(list(ys), color=INK); a.tick_params(length=0)
        for sp in a.spines.values(): sp.set_visible(False)
        a.set_title({'PrAl': 'Pr-Al (500 + 500 genes)', 'axis1': 'Axis 1 (100 + 100 genes)', 'axis2': 'Axis 2 (100 + 100 genes)', 'axis3': 'Axis 3 (300 + 300 genes)', 'axis4': 'Axis 4 (500 + 500 genes)', 'axis5': 'Axis 5 (500 + 500 genes)'}.get(s, s) + f'\ncolour scale up to {vmax:.1f}', loc='left', fontsize=11, fontweight='bold', color=INK, pad=24)
    fig.suptitle('PAD by subclass and age (x 1000, diagonal excluded)', x=0.02, ha='left', fontsize=13, fontweight='bold', color=INK, y=0.985)
    fig.text(0.02, 0.012, 'PAD = mean gene-gene r within the two sets minus between them; Pearson over nuclei of the subclass, log1p CP10k, all cortical samples pooled. Positive PAD is not proof of opposition (see between-set block).\n'
             'One donor per young age, adult donor not given; regions differ between ages; depth and number of nuclei not matched. Colour scales differ between panels; orange = negative. * fewer than 500 nuclei.', fontsize=8, color=INK2, va='bottom')
    fig.tight_layout(rect=(0, 0.05, 1, 0.97)); fig.savefig(cfg().pictures(name), dpi=150)


if __name__ == '__main__':
    import sys
    if '--plot' in sys.argv: plot(); plot(('axis2', 'axis3', 'axis4', 'axis5'), 'snrna_pad_heatmap_axes2_5.png'); sys.exit()
    run()
