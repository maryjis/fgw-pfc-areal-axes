"""Heatmap of the snRNA axis index z(+ end) - z(- end): subclass x adult sample (table snrna_types/scores_samples.csv).
python -m ottr.genescore.snrna_heatmap  -> results/pictures/snrna_index_heatmap.png (PrAl, axes 1, 3), snrna_index_heatmap_axes245.png"""
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from ..config import cfg
from . import snrna_types as ST

PFC = ['A47', 'A8daD', 'A9A46', 'A10', 'A13', 'A32', 'A24A32']; REST = ['A3A4', 'V1', 'V23', 'AuA1', 'AuR', 'pTE']
BLOCKS = [['L2/3 IT + ExN_dev', 'L2/3 IT (no ExN_dev)', 'L4 IT', 'L5 IT', 'L6 IT', 'L6 CT', 'L5 ET'], ['Pvalb', 'Sst', 'Vip', 'Lamp5'], ['Astro', 'Oligo', 'OPC']]
NAMES = {'PrAl': 'Pr-Al', **{f'axis{k}': f'Axis {k}' for k in range(1, 6)}}; INK, INK2, NA = '#14161a', '#5b5e66', '#ececea'
CM = LinearSegmentedColormap.from_list('div', ['#1c5fb0', '#8db9e8', '#f1f0ee', '#f3ab86', '#c9480f'])


def draw(scores, name):
    S = pd.read_csv(os.path.join(ST.out_dir(), 'scores_samples.csv')); T = ST.targets(); xs = {r: i for i, r in enumerate(PFC)}; xs.update({r: len(PFC) + .4 + i for i, r in enumerate(REST)})
    ys, y = {'Stereo-seq': 0}, 1.5
    for b in BLOCKS:
        for g in b: ys[g] = y; y += 1
        y += .4
    fig, ax = plt.subplots(1, len(scores), figsize=(6.7 * len(scores) + 1.8, 8.2), facecolor='white', sharey=True)
    for a, s in zip(ax, scores):
        V = S[S.score == s].pivot(index='group', columns='region_label', values='mean'); N = S[S.score == s].pivot(index='group', columns='region_label', values='n_nuclei'); vm = float(np.nanquantile(np.abs(V.values), .97))
        st = T[s, 'stereo']; sv = float(np.abs(st - st.mean()).max())
        def cell(x, yy, v, f, txt):
            c = CM(.5 + .5 * np.clip(f, -1, 1)); a.add_patch(plt.Rectangle((x + .03, yy + .03), .94, .94, color=c)); a.text(x + .5, yy + .5, txt, ha='center', va='center', fontsize=8, color='white' if abs(f) > .75 else INK)
        for r in PFC: cell(xs[r], 0, st[r], (st[r] - st.mean()) / sv, f'{st[r]:.2f}')
        for r in REST: a.add_patch(plt.Rectangle((xs[r] + .03, .03), .94, .94, color=NA))
        for g in [k for k in ys if k != 'Stereo-seq']:
            for r, x in xs.items():
                v = V.loc[g, r] if r in V.columns else np.nan; n = N.loc[g, r] if r in N.columns else 0
                if np.isfinite(v): cell(x, ys[g], v, v / vm, f'{v:.1f}' + ('*' if n < ST.MIN_N else ''))
                else: a.add_patch(plt.Rectangle((x + .03, ys[g] + .03), .94, .94, color=NA))
        a.set_xlim(0, max(xs.values()) + 1); a.set_ylim(y - .3, -.1); a.set_xticks([x + .5 for x in xs.values()]); a.set_xticklabels(list(xs), rotation=90, color=INK, fontsize=9); a.xaxis.tick_top()
        a.set_yticks([v + .5 for v in ys.values()]); a.set_yticklabels(list(ys), color=INK, fontsize=9.5); a.tick_params(length=0)
        for sp in a.spines.values(): sp.set_visible(False)
        a.set_title(f'{NAMES[s]}: z(+) - z(-), colour limit {vm:.1f}', loc='left', fontsize=11, fontweight='bold', color=INK, pad=10)
    fig.suptitle('snRNA index of the axes by subclass and region, adult (left 7 columns: PFC; right 6: other cortex)', x=.02, ha='left', fontsize=13, fontweight='bold', color=INK, y=.985)
    fig.text(.02, .01, 'Index of a nucleus = z(mean of + end genes) - z(mean of - end genes), z within the subclass (all 13 adult samples), averaged per sample: 0 = mean of the subclass, so rows compare in shape, not in level. Orange = + end.\n'
             'Top row: Stereo-seq score of the same PFC regions (own colour scale, centred on its mean). * fewer than 100 nuclei (not used in correlations). One sample per region; adult donor not given; depth and number of nuclei not matched.', fontsize=8, color=INK2, va='bottom')
    fig.tight_layout(rect=(0, .07, 1, .97)); fig.savefig(cfg().pictures(name), dpi=150)


def draw_sorted(scores, name):
    """Columns sorted by the mean index over the subclasses (L2/3 counted once, cells with < MIN_N nuclei left out; own order in every
    panel), rows within a class sorted by the range between regions (regions with >= MIN_N nuclei), range printed on the right."""
    S = pd.read_csv(os.path.join(ST.out_dir(), 'scores_samples.csv')); T = ST.targets(); allr = PFC + REST; tab = []
    fig, ax = plt.subplots(1, len(scores), figsize=(8.6 * len(scores), 8.4), facecolor='white')
    for a, s in zip(ax, scores):
        V = S[S.score == s].pivot(index='group', columns='region_label', values='mean')[allr]; N = S[S.score == s].pivot(index='group', columns='region_label', values='n_nuclei')[allr]
        K = V.where(N >= ST.MIN_N); order = list(K.drop(index='L2/3 IT (no ExN_dev)').mean().sort_values(ascending=False).index); rng_ = K.max(axis=1) - K.min(axis=1); vm = float(np.nanquantile(np.abs(V.values), .97))
        st = T[s, 'stereo']; sv = float(np.abs(st - st.mean()).max()); ys, y = {'Stereo-seq (PFC)': 0}, 1.5
        for b in BLOCKS:
            for g in sorted(b, key=lambda g: -rng_[g]): ys[g] = y; y += 1
            y += .4
        def cell(x, yy, f, txt):
            a.add_patch(plt.Rectangle((x + .03, yy + .03), .94, .94, color=CM(.5 + .5 * np.clip(f, -1, 1)))); a.text(x + .5, yy + .5, txt, ha='center', va='center', fontsize=8, color='white' if abs(f) > .75 else INK)
        for i, r in enumerate(order):
            if r in PFC: cell(i, 0, (st[r] - st.mean()) / sv, f'{st[r]:.2f}')
            else: a.add_patch(plt.Rectangle((i + .03, .03), .94, .94, color=NA))
            for g, yy in list(ys.items())[1:]: cell(i, yy, V.loc[g, r] / vm, f'{V.loc[g, r]:.1f}' + ('*' if N.loc[g, r] < ST.MIN_N else ''))
        for g, yy in list(ys.items())[1:]: a.text(len(order) + .35, yy + .5, f'{rng_[g]:.1f}', va='center', fontsize=9, color=INK); tab.append(dict(score=s, group=g, range_13=rng_[g], order=','.join(order)))
        a.text(len(order) + .35, 1.15, 'range', fontsize=8.5, color=INK2, va='center'); a.set_xlim(0, len(order) + 1.3); a.set_ylim(y - .3, -.1); a.set_xticks(np.arange(len(order)) + .5)
        a.set_xticklabels(order, rotation=90, fontsize=9, color=INK2); a.xaxis.tick_top()
        for tl, r in zip(a.get_xticklabels(), order):
            if r in PFC: tl.set_fontweight('bold'); tl.set_color(INK)
        a.set_yticks([v + .5 for v in ys.values()]); a.set_yticklabels(list(ys), color=INK, fontsize=9.5); a.tick_params(length=0)
        for sp in a.spines.values(): sp.set_visible(False)
        a.set_title(f'{NAMES[s]}: colour limit {vm:.1f}', loc='left', fontsize=11, fontweight='bold', color=INK, pad=10)
    fig.suptitle('snRNA index z(+) - z(-), adult: regions ordered by the index, subclasses by the range between regions', x=.02, ha='left', fontsize=13, fontweight='bold', color=INK, y=.985)
    fig.text(.02, .01, 'Columns: from the + end to the - end by the mean index over the subclasses (own order in every panel; bold = PFC). The order comes from the same values that are drawn, so a smooth gradient is partly by construction;\n'
             'the independent check is the Stereo-seq row (PFC only, own colour scale). Rows compare in shape, not in level; range = max - min over the samples with >= 100 nuclei (* = fewer), depth and number of nuclei not matched. One sample per region.', fontsize=8, color=INK2, va='bottom')
    fig.tight_layout(rect=(0, .07, 1, .97)); fig.savefig(cfg().pictures(name), dpi=150); return pd.DataFrame(tab)


if __name__ == '__main__':
    t = pd.concat([draw_sorted(('PrAl', 'axis1', 'axis3'), 'snrna_index_heatmap_sorted.png'), draw_sorted(('axis2', 'axis4', 'axis5'), 'snrna_index_heatmap_sorted_axes245.png')])
    t.to_csv(os.path.join(ST.out_dir(), 'index_range.csv'), index=False)
    draw(('PrAl', 'axis1', 'axis3'), 'snrna_index_heatmap.png'); draw(('axis2', 'axis4', 'axis5'), 'snrna_index_heatmap_axes245.png')
