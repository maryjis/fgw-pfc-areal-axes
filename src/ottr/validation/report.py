"""E11 of the validation plan: master table of every comparison and the axis x domain summary.   python -m ottr.validation.report
-> results/tables/validation/master_table.csv, master_table.xlsx (sheets: primary, all, summary, legend; q < 0.05 highlighted per null),
   summary_matrix.csv, results/pictures/validation_summary.png"""
import os, glob
import numpy as np
import pandas as pd
from ..config import cfg
from . import common as C

ORDER = ['snRNA subclasses', 'atlas maps', 'atlas components (E2)', 'connectivity', 'morphometry', 'cell-type composition', 'species', 'species components (E3)',
         'macaque composition vs expression (E4)', 'human snRNA (Siletti 2023)']
RANK = {'confirmed': 3, 'nominal': 2, 'perm only': 1, 'no': 0}


def domain_label(r):
    """finer domain for the summary: the atlas maps split by modality, species split by species."""
    if r.domain == 'atlas maps': return f'atlas: {r.dataset}'
    if r.domain in ('species', 'species components (E3)', 'macaque composition vs expression (E4)'): return f'{r.domain} [{r.species}]'
    return r.domain


def load():
    fs = [f for f in sorted(glob.glob(os.path.join(C.out_dir(), 'rows_*.csv'))) if not f.endswith(('rows_snrna_pca.csv', 'rows_residual_pca.csv'))]   # residual_pca: composition removed, not the model's analogue (kept for the record)
    D = pd.concat([pd.read_csv(f).assign(source=os.path.basename(f)) for f in fs], ignore_index=True); D['domain_summary'] = D.apply(domain_label, axis=1)
    D['axis'] = D.axis.astype(str).str.replace('.0', '', regex=False); return D


def summary(D):
    P = D[(D.family == 'primary') & D.axis.isin(['1', '2', '3', '4', '5'])]; out = []
    for (dm, k), g in P.groupby(['domain_summary', 'axis']):
        g = g.assign(rank=g.verdict.map(RANK)).sort_values(['rank', 'p_msr'], ascending=[False, True]); b = g.iloc[0]
        out.append(dict(domain=dm, axis=k, n_tests=len(g), best_feature=f'{b.dataset} | {b.feature}', rho=b.spearman, q_perm=b.q_perm, q_msr=b.q_msr, q_vario=b.q_vario, verdict=b.verdict,
                        n_confirmed=int((g.verdict == 'confirmed').sum()), n_perm_sig=int((g.q_perm < .05).sum())))
    return pd.DataFrame(out)


def xlsx(D, S, path):
    from openpyxl.styles import PatternFill, Font
    cols = ['domain', 'dataset', 'species', 'axis', 'ours', 'feature', 'family', 'n', 'spearman', 'pearson', 'ci_low', 'ci_high', 'p_perm', 'q_perm', 'p_msr', 'q_msr', 'p_vario', 'q_vario',
            'p_extra', 'p_extra_kind', 'replication', 'loo_min', 'loo_max', 'verdict', 'note']
    fill = {'q_perm': PatternFill('solid', fgColor='FCE4D6'), 'q_msr': PatternFill('solid', fgColor='C6E0B4'), 'q_vario': PatternFill('solid', fgColor='C6E0B4')}
    vfill = {'confirmed': PatternFill('solid', fgColor='A9D08E'), 'nominal': PatternFill('solid', fgColor='E2EFDA'), 'perm only': PatternFill('solid', fgColor='FCE4D6')}
    legend = pd.DataFrame({'column': ['p_perm / q_perm', 'p_msr / q_msr', 'p_vario / q_vario', 'p_extra', 'verdict', 'family', 'replication'],
                           'meaning': ['plain permutation of the units (exact for n <= 8); q = BH within domain x family',
                                       'spatial null: Moran spectral randomisation surrogates of OUR map (30 regions), carried to the units; q = BH',
                                       'spatial null: variogram-matched surrogates; q = BH',
                                       'additional null where available (gene-set null)',
                                       'confirmed: q_msr and q_vario < 0.05, sign stable in leave-one-out, replication not opposite; nominal: p_msr and p_vario < 0.05; perm only: q_perm < 0.05 only',
                                       'primary tests were fixed before the run; exploratory are corrected within their own family',
                                       'other animal / outside our series / other donors: same or opposite sign']})
    with pd.ExcelWriter(path, engine='openpyxl') as W:
        for name, T in (('primary', D[D.family == 'primary'][cols]), ('all', D[cols]), ('summary', S), ('legend', legend)):
            T.to_excel(W, sheet_name=name, index=False); ws = W.sheets[name]
            if name in ('primary', 'all', 'summary'):
                hdr = [c.value for c in ws[1]]
                for row in ws.iter_rows(min_row=2):
                    for cell, h in zip(row, hdr):
                        if h in fill and isinstance(cell.value, (int, float)) and cell.value < .05: cell.fill = fill[h]; cell.font = Font(bold=True)
                        if h == 'verdict' and cell.value in vfill: cell.fill = vfill[cell.value]
                        if isinstance(cell.value, float): cell.number_format = '0.0000' if h.startswith(('p_', 'q_')) else '0.00'
            for col in ws.columns: ws.column_dimensions[col[0].column_letter].width = min(60, max(9, max(len(str(c.value or '')) for c in col[:200]) * 0.9))
            ws.freeze_panes = 'B2'


def plot(S):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    INK, INK2 = '#14161a', '#5b5e66'; COL = {'confirmed': '#2e7d32', 'nominal': '#a5d6a7', 'perm only': '#f6c9a8', 'no': '#eeeeec'}
    doms = sorted(S.domain.unique(), key=lambda d: [i for i, o in enumerate(ORDER) if d.startswith(o.split(' (')[0].split(':')[0])][:1] or [99]); axes = ['1', '2', '3', '4', '5']
    fig, ax = plt.subplots(figsize=(13, 0.5 * len(doms) + 2.2), facecolor='white')
    for i, dm in enumerate(doms):
        for j, k in enumerate(axes):
            r = S[(S.domain == dm) & (S.axis == k)]
            if r.empty: ax.add_patch(plt.Rectangle((j + .04, i + .06), .92, .88, fill=False, ec='#d5d4cf', ls=(0, (2, 2)))); ax.text(j + .5, i + .5, 'no primary test', ha='center', va='center', fontsize=8, color=INK2); continue
            r = r.iloc[0]; ax.add_patch(plt.Rectangle((j + .04, i + .06), .92, .88, color=COL[r.verdict]))
            ax.text(j + .5, i + .5, f'ρ {r.rho:+.2f}  q {r.q_msr:.2g}/{r.q_vario:.2g}\nperm q {r.q_perm:.2g}', ha='center', va='center', fontsize=7.6, color='white' if r.verdict == 'confirmed' else INK)
    ax.set_xlim(0, 5); ax.set_ylim(len(doms), 0); ax.set_xticks(np.arange(5) + .5); ax.set_xticklabels([f'Axis {k}' for k in axes], fontsize=10, color=INK); ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(doms)) + .5); ax.set_yticklabels(doms, fontsize=9, color=INK); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()]; ax.legend(hs, ['confirmed (spatial q < 0.05 both, LOO, replication)', 'nominal (spatial p < 0.05 both)', 'permutation q < 0.05 only', 'none'],
                                                                              loc='upper center', bbox_to_anchor=(.5, -.02), ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle('Primary tests: best result per axis and domain (ρ; q spatial Moran / variogram; q plain permutation)', x=.01, ha='left', fontsize=12, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, .03, 1, .95)); fig.savefig(cfg().pictures('validation_summary.png'), dpi=150)


def run():
    D = load(); S = summary(D); o = C.out_dir(); D.to_csv(os.path.join(o, 'master_table.csv'), index=False); S.to_csv(os.path.join(o, 'summary_matrix.csv'), index=False)
    xlsx(D, S, os.path.join(o, 'master_table.xlsx')); plot(S); return D, S


if __name__ == '__main__':
    pd.set_option('display.width', 250); D, S = run(); print(D.groupby(['domain', 'family', 'verdict']).size().unstack(fill_value=0).to_string()); print(S.round(3).to_string(index=False))
