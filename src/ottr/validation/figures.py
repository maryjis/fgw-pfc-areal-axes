"""Heatmaps of the validation rows for the final report: axes 1-5 x features, Spearman rho in the cell, verdict marked
(** confirmed: q < 0.05 by both spatial nulls + LOO + replication; * nominal: p < 0.05 by both spatial nulls; ° plain permutation q < 0.05 only).
python -m ottr.validation.figures  ->  results/pictures/validation_heat_{atlas,morphometry,connectivity,snrna,species}.png"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from ..config import cfg
from . import common as C

INK, INK2, GRID = '#1f1f1f', '#5b5e66', '#d9d9d6'
CMAP = LinearSegmentedColormap.from_list('div', ['#0f4a96', '#519ced', '#efefef', '#eb6263', '#a8282a'])
MARK = {'confirmed': '**', 'nominal': '*', 'perm only': '°', 'no': ''}
AXES = ['1', '2', '3', '4', '5']


def rows(name):
    D = pd.read_csv(os.path.join(C.out_dir(), f'rows_{name}.csv')); D['axis'] = D.axis.astype(str).str.replace('.0', '', regex=False); return D


def heat(cols, title, sub, out, groups=None):
    """cols: list of (label, {axis: row or None}). groups: list of (label, n_columns) drawn as headers above the columns."""
    n = len(cols); w = max(10.5, 0.62 * n + 2.2); fig, ax = plt.subplots(figsize=(w, 4.3 if groups else 4.0), facecolor='white')
    for j, (lab, cells) in enumerate(cols):
        for i, k in enumerate(AXES):
            r = cells.get(k)
            if r is None: ax.add_patch(plt.Rectangle((j + .03, i + .03), .94, .94, fc='white', ec=GRID, lw=.6)); continue
            v = float(r['spearman']); col = CMAP((v + 1) / 2); ax.add_patch(plt.Rectangle((j + .03, i + .03), .94, .94, fc=col, ec='white', lw=1.5))
            vd = r['verdict']
            if vd == 'confirmed': ax.add_patch(plt.Rectangle((j + .06, i + .06), .88, .88, fill=False, ec=INK, lw=2.2))
            elif vd == 'nominal': ax.add_patch(plt.Rectangle((j + .06, i + .06), .88, .88, fill=False, ec=INK, lw=1.0))
            tc = 'white' if abs(v) > .62 else INK
            ax.text(j + .5, i + .5, f'{v:+.2f}'.replace('.', ',') + MARK.get(vd, ''), ha='center', va='center', fontsize=7.6, color=tc, fontweight='bold' if vd in ('confirmed', 'nominal') else 'normal')
    ax.set_xlim(0, n); ax.set_ylim(5, 0); ax.set_yticks(np.arange(5) + .5); ax.set_yticklabels([f'Ось {k}' for k in AXES], fontsize=9.5, color=INK)
    ax.set_xticks(np.arange(n) + .5); ax.set_xticklabels([c[0] for c in cols], rotation=55, ha='right', fontsize=8.5, color=INK); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    if groups:
        x = 0
        for lab, m in groups:
            ax.annotate('', xy=(x + .1, -.25), xytext=(x + m - .1, -.25), arrowprops=dict(arrowstyle='-', color=INK2, lw=.8), annotation_clip=False)
            ax.text(x + m / 2, -.4, lab, ha='center', va='bottom', fontsize=8.5, color=INK2); x += m
    fig.text(.01, .97, title, ha='left', va='top', fontsize=12, fontweight='bold', color=INK)
    fig.text(.01, .915, sub, ha='left', va='top', fontsize=8, color=INK2); fig.text(.01, .875, '** подтверждено (жирная рамка) · * номинально по обоим пространственным нулям (тонкая рамка) · ° только перестановки', ha='left', va='top', fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, .80 if groups else .84)); p = cfg().pictures(out); fig.savefig(p, dpi=170); plt.close(fig); return p


def by_axis(D, **sel):
    q = D.copy()
    for k, v in sel.items(): q = q[q[k] == v]
    return {r['axis']: r for _, r in q.iterrows()}


def atlas():
    D = rows('atlas'); D = D[D.ours.str.startswith('mu') & ~D.feature.str.startswith('best of')]
    mri = [('Myelinmap', 'Миелин'), ('thickness_L2-L6', 'Толщина L2–6'), ('L4_fraction', 'Доля L4')] + [(f'L{l}-Layer-Thickness', f'Толщина L{l}') for l in range(2, 7)]
    grad = [('FG1', 'FG1'), ('FG2', 'FG2'), ('SG1', 'SG1'), ('SG2', 'SG2'), ('Hierachical', 'Иерархия')]
    nets = sorted(f for f in D.feature.unique() if f.endswith('-Network')); tasks = [f for f in D.feature.unique() if f.endswith('-Task')]
    cols = [(lab, by_axis(D[D.dataset == 'MRI'], feature=f)) for f, lab in mri]
    cols += [(lab, by_axis(D[D.dataset.isin(['fMRI', 'connectivity', 'hierarchy'])].drop_duplicates(['feature', 'axis']), feature=f)) for f, lab in grad]
    cols += [(f.replace('-Network', ''), by_axis(D, feature=f)) for f in nets] + [(f.replace('-Task', ' (задача)'), by_axis(D, feature=f)) for f in tasks]
    return heat(cols, 'E1. Оси против карт атласа MRI и fMRI (30 областей PFC)', 'Spearman ρ; первичные: миелин, FG1–2, SG1–2; остальное разведочно.', 'validation_heat_atlas.png',
                groups=[('MRI: миелин и слои', len(mri)), ('градиенты', len(grad)), ('сети fMRI', len(nets)), ('задачи', len(tasks))])


def morphometry():
    B = rows('morphometry'); B = B[B.ours.str.startswith('mu')]; M = rows('morphometry_mbm'); M = M[M.ours.str.startswith('mu')]
    cb = [('thickness', 'Толщина'), ('sulcal_depth', 'Глубина борозд'), ('surface_area', 'Площадь'), ('mean_curvature', 'Кривизна')]
    cm = [('thickness', 'Толщина'), ('sulcal_depth', 'Глубина борозд'), ('surface_area', 'Площадь'), ('pial_curv', 'Кривизна pial'), ('white_curv', 'Кривизна white'), ('myelin', 'Миелин')]
    cols = [(lab, by_axis(B, feature=f)) for f, lab in cb] + [(lab, by_axis(M, feature=f)) for f, lab in cm]
    return heat(cols, 'E10. Морфометрия: BMA2.0 и MRI-шаблон MBM', 'Spearman ρ; 30 / 28 областей; все тесты разведочные.', 'validation_heat_morphometry.png',
                groups=[('BMA2.0 (наш расчёт)', len(cb)), ('MBM (карты авторов + наш расчёт)', len(cm))])


def connectivity():
    D = rows('connectivity'); D = D[D.ours.str.startswith('mu')]
    sel = [('FC (MBMv4 resting state)', 'Mantel: profile dissimilarity', 'FC Mantel'), ('SC (MBMv4 dMRI tractography)', 'Mantel: profile dissimilarity', 'SC Mantel'),
           ('FLN outputs (Majka 2020 tracing)', 'Mantel: profile dissimilarity', 'FLN выходы Mantel'), ('FLN inputs of injected PFC regions (Majka 2020)', 'Mantel: input profile dissimilarity', 'FLN входы Mantel (14)')]
    sel += [(f'{m} ({d})', f'diffusion gradient G{g} (whole cortex)', f'{m.split()[0]} G{g}') for m, d in (('FC', 'MBMv4 resting state'), ('SC', 'MBMv4 dMRI tractography')) for g in (1, 2, 3)]
    cols = [(lab, by_axis(D, dataset=ds, feature=f)) for ds, f, lab in sel]
    return heat(cols, 'E7–E8. Связность: FC, SC (MBMv4) и трассировка FLN (Majka 2020)', 'Mantel: ρ по парам областей; градиенты: ρ по 30 областям, знак произвольный. Первичные: Mantel FC, SC, FLN выходы.',
                'validation_heat_connectivity.png', groups=[('Mantel', 4), ('градиенты (diffusion map)', 6)])


def snrna():
    D = rows('snrna'); D = D[D.ours.str.startswith('mu')]; D['g'] = D.dataset.str.replace('adult snRNA-seq, ', '', regex=False)
    order = ['L2/3 IT + ExN_dev', 'L4 IT', 'L5 IT', 'L6 IT', 'L5 ET', 'L6 CT', 'Pvalb', 'Sst', 'Vip', 'Lamp5', 'Astro', 'Oligo', 'OPC']
    cols = [(g, by_axis(D, g=g)) for g in order]
    return heat(cols, 'E5. snRNA мармозетки: скор оси в подклассе против μ (7 проб PFC)', 'Spearman ρ по пробам; все подклассы из одних 7 проб (L2/3 IT без ExN_dev совпадает с L2/3 IT + ExN_dev).',
                'validation_heat_snrna.png', groups=[('возбуждающие', 6), ('тормозные', 4), ('глия', 3)])


def species():
    S = rows('species'); S = S[S.feature.str.contains('axis')]; E3 = rows('species_pca'); E4 = rows('e4_macaque'); E4 = E4[~E4.feature.str.startswith('PrAl')]
    ML = rows('model_like_pca'); SI = rows('siletti'); SI = SI[SI.ours.str.startswith('mu') & SI.feature.str.startswith('axis')]
    SI = SI[SI.apply(lambda r: r.feature.split()[0] == f'axis{r.axis}', axis=1)]
    cols = [('Гены макаки', by_axis(S, species='macaque')), ('PCA лобных, макака', by_axis(E3, dataset='macaque frontal PCA')), ('PCA всей коры, макака', by_axis(E3, dataset='macaque whole cortex PCA')),
            ('PCA лобных − слои, макака', by_axis(ML, dataset='macaque frontal PCA, layer shares L1-L5 + depth removed'))]
    cols += [(f'Гены | состав R{i}', by_axis(E4, dataset=f'macaque gene score | Cell-PC1-3 of R{i}')) for i in (1, 2, 3)]
    cols += [('Гены человека', by_axis(S, species='human')), ('PCA лобных, человек', by_axis(E3, dataset='human frontal PCA')), ('PCA всей коры, человек', by_axis(E3, dataset='human whole cortex PCA'))]
    cols += [(f'Siletti {sc.split("-")[0].lower()} IT', by_axis(SI, dataset=sc)) for sc in ('Upper-layer intratelencephalic', 'Deep-layer intratelencephalic')]
    return heat(cols, 'Другие виды: гены осей, компоненты видов, учёт состава, Siletti', 'Spearman ρ; для PCA — лучшая из PC1–3 (PC1–5 после вычитания слоёв), знак произвольный. n: макака 35, человек 29, Siletti 8.',
                'validation_heat_species.png', groups=[('макака', 7), ('человек', 3), ('snRNA человека', 2)])


if __name__ == '__main__':
    for f in (atlas, morphometry, connectivity, snrna, species): print(f())
