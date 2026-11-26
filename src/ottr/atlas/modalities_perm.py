"""E1 / E2 with the plain permutation test (regions shuffled, 10 000 permutations) instead of the spatial nulls.
python -m ottr.atlas.modalities_perm  -> results/tables/validation/{e1_regions_perm,e2_match_perm}.csv, results/pictures/validation_e1_perm*.png
BH families as in modalities: primary (mu x Myelinmap/FG1/FG2/SG1/SG2, 25 tests), exploratory mu (130), gene scores (all 186), E2 (15).
Plain permutations ignore that neighbouring regions are alike: on independent smooth random maps of these 30 regions they give
p < 0.05 in 4-13% of cases (modalities.null_check) instead of 5%."""
import os
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from ..config import cfg, ensure
from ..genescore import significance as SG
from . import structure as ST, modalities as MD

NP = 10000
SHORT = {'Myelinmap': 'Myelin', 'Hierachical': 'Hierarchy', 'thickness_L2-L6': 'Thick L2-6', 'L4_fraction': 'L4 fraction', **{f'{l}-Layer-Thickness': f'{l} thick' for l in ST.LAYERS},
         **{f'{n}-Network': n for n in MD.NET}, 'Animation-Task': 'Task anim', 'Vocal-Task': 'Task vocal', 'Visual-Task': 'Task visual'}


def bh(p): return multipletests(p, method='fdr_bh')[1]


def run():
    c = cfg(); out = ensure(c.tables('validation')); rng = np.random.default_rng(41); M = ST.meta(); O, A, B = MD.region_table(M); n = len(O)
    P = np.array([rng.permutation(n) for _ in range(NP)]); rows = []
    for o in O:
        x = O[o].values.astype(float); rx = SG.unit(x); Rp = SG.unit(x)[P]                        # permuted ranks of ours
        for f in MD.FEATS:
            y = A[f].values.astype(float); u = SG.unit(y); r = float(rx @ u); null = Rp @ u
            rows.append(dict(ours=o, feature=f, family='primary' if (o.startswith('mu') and f in MD.PRIMARY) else ('exploratory mu' if o.startswith('mu') else 'gene scores'),
                             n=n, spearman=r, pearson=float(np.corrcoef(x, y)[0, 1]), p_perm=(1 + (np.abs(null) >= abs(r) - 1e-12).sum()) / (NP + 1)))
    D = pd.DataFrame(rows); D['q_perm'] = np.nan
    for _, i in D.groupby('family').groups.items(): D.loc[i, 'q_perm'] = bh(D.loc[i, 'p_perm'])
    S = pd.read_csv(os.path.join(out, 'e1_regions.csv'))[['ours', 'feature', 'q_msr', 'q_variogram']]; D = D.merge(S, on=['ours', 'feature'], how='left'); D.to_csv(os.path.join(out, 'e1_regions_perm.csv'), index=False)
    # E2: best component per modality, null = max over components under the same permutations
    comp = {}
    for mod, fs in MD.MODALITY.items():
        X = A[fs].values.astype(float)
        if mod == 'MRI': Z = (X - X.mean(0)) / X.std(0); U, s, _ = np.linalg.svd(Z, full_matrices=False); comp[mod] = pd.DataFrame(U[:, :3] * s[:3], index=A.index, columns=['MRI-PC1', 'MRI-PC2', 'MRI-PC3'])
        else: comp[mod] = A[fs]
    e2 = []
    for k in range(1, 6):
        x = O[f'mu axis{k}'].values.astype(float); Rp = SG.unit(x)[P]
        for mod, T in comp.items():
            R = np.array([SG.rho(x, T[j].values) for j in T]); N = np.abs(np.stack([Rp @ SG.unit(T[j].values) for j in T])).max(0); b = int(np.abs(R).argmax())
            e2.append(dict(axis=k, modality=mod, best=T.columns[b], rho=R[b], p_perm=(1 + (N >= abs(R[b]) - 1e-12).sum()) / (NP + 1)))
    E = pd.DataFrame(e2); E['q_perm'] = bh(E.p_perm); E.to_csv(os.path.join(out, 'e2_match_perm.csv'), index=False)
    plot(D, E); return D, E


def plot(D, E):
    INK, INK2 = '#14161a', '#5b5e66'; CM = LinearSegmentedColormap.from_list('div', ['#1c5fb0', '#8db9e8', '#f1f0ee', '#f3ab86', '#c9480f']); c = cfg()
    groups = [('MRI and layers', ['Myelinmap', 'thickness_L2-L6', 'L4_fraction'] + [f'{l}-Layer-Thickness' for l in ST.LAYERS]), ('fMRI gradients, connectivity, hierarchy', ['FG1', 'FG2', 'SG1', 'SG2', 'Hierachical']),
              ('fMRI networks', [f'{n}-Network' for n in MD.NET]), ('Tasks', ['Animation-Task', 'Vocal-Task', 'Visual-Task'])]
    rowsets = [('mu', [f'mu axis{k}' for k in range(1, 6)]), ('genes', ['genes PrAl'] + [f'genes axis{k}' for k in range(1, 6)])]
    cols = [f for _, fs in groups for f in fs]; xs, x = [], 0
    for _, fs in groups:
        for f in fs: xs.append(x); x += 1
        x += .5
    fig, ax = plt.subplots(2, 1, figsize=(19, 8.6), facecolor='white', gridspec_kw=dict(height_ratios=[5, 6]))
    for a, (kind, rs) in zip(ax, rowsets):
        for i, o in enumerate(rs):
            for xx, f in zip(xs, cols):
                d = D[(D.ours == o) & (D.feature == f)].iloc[0]; prim = d.family == 'primary'; sig = d.q_perm < .05
                a.add_patch(plt.Rectangle((xx + .04, i + .04), .92, .92, color=CM(.5 + .5 * d.spearman)))
                if sig: a.add_patch(plt.Rectangle((xx + .06, i + .06), .88, .88, fill=False, lw=2.2, ec=INK))
                if prim: a.add_patch(plt.Rectangle((xx + .02, i + .02), .96, .96, fill=False, lw=.9, ec=INK2, ls=(0, (2, 2))))
                a.text(xx + .5, i + .5, f'{d.spearman:.2f}' + ('*' if sig else ''), ha='center', va='center', fontsize=7.6, fontweight='bold' if sig else 'normal', color='white' if abs(d.spearman) > .7 else INK)
        a.set_xlim(0, x - .5); a.set_ylim(len(rs), 0); a.set_yticks(np.arange(len(rs)) + .5); a.set_yticklabels([r.replace('mu axis', 'Axis ').replace('genes ', 'genes ') + (' μ' if kind == 'mu' else '') for r in rs], fontsize=9.5, color=INK)
        a.set_xticks(np.array(xs) + .5); a.set_xticklabels([SHORT.get(f, f) for f in cols], rotation=90, fontsize=8.5, color=INK) if kind == 'genes' else a.set_xticks([])
        if kind == 'mu':
            for (g, fs) in groups: a.text(xs[cols.index(fs[0])], -.25, g, fontsize=9, fontweight='bold', color=INK)
        a.tick_params(length=0)
        for sp in a.spines.values(): sp.set_visible(False)
        a.set_title(('Region score μ of the axis' if kind == 'mu' else 'Gene-set score of the axis (Stereo-seq, adult reference)'), loc='left', fontsize=10.5, fontweight='bold', color=INK, pad=22 if kind == 'mu' else 6)
    fig.suptitle('Our axes vs the maps on the atlas segments, 30 PFC regions: Spearman ρ; framed and * = q < 0.05 (plain permutation test, BH within family)', x=.01, ha='left', fontsize=12.5, fontweight='bold', color=INK)
    fig.text(.01, .005, 'Families for BH: primary = μ x {Myelin, FG1, FG2, SG1, SG2} (25 tests, dashed outline); exploratory μ (130); gene scores (186). Plain permutations treat regions as independent; neighbouring regions are alike,\n'
             'so p is too small (on independent smooth random maps of these regions: p < 0.05 in 4-13% instead of 5%). With the spatial nulls only Axis 1 x Myelin passes. Signs of FG/SG/networks are as given by the authors.', fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, .05, 1, .95)); fig.savefig(c.pictures('validation_e1_perm.png'), dpi=150)


if __name__ == '__main__':
    D, E = run(); pd.set_option('display.width', 250)
    for fam, d in D.groupby('family'): print('##', fam, len(d), 'q<.05:', (d.q_perm < .05).sum()); print(d[d.q_perm < .05].sort_values('p_perm')[['ours', 'feature', 'spearman', 'pearson', 'p_perm', 'q_perm', 'q_msr', 'q_variogram']].round(4).to_string(index=False))
    print(D[D.family == 'primary'].pivot(index='ours', columns='feature', values='q_perm').round(4).to_string()); print(E.round(4).to_string())
