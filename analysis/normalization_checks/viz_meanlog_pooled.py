"""Mean expression over all genes: mean of logs vs log of sums (both normalised by the spot's library size).
  per spot i, gene g:  y_ig = log1p(1e4 x_ig / L_i),  L_i = sum of the counts of the 1937 model genes
  mean of logs  over a group G : mean_g mean_{i in G} y_ig
  log of sums   over a group G : mean_g log1p(1e4 sum_{i in G} x_ig / sum_{i in G} L_i)
Row 1: one point per region (all L1-L6 spots of the region), centred over regions, vs the region's mean UMI per spot;
       SimR (random counts, real per-spot totals: the truth is the same value everywhere) and the real series.
Row 2: one section of SimR on the tissue: UMI per spot (disk mean r 1.5) and the two estimates over the same disk,
       as deviation from the section mean (shared colour scale).
-> pooled/meanlog_vs_pooled.png"""
import sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, '/workspace/Gene_modules/optimal_transport__tr_axeses/src')
from ottr.utils.smoothing import grid_positions, disk_mean

W = '/workspace/Gene_modules/optimal_transport__tr_axeses/work/inputs'; OUT = sys.argv[1]; SEC = 'T419'
BLUE, ORANGE, INK, INK2, GRID, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SEQ = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f19a99', '#b52a29'])
LAY = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']; BAD = ['nan', 'None', 'remove', 'unassigned', 'WM', 'white_matter']


def load(ds):
    A = ad.read_h5ad(f'{W}/{ds}.h5ad'); Y = A.X.tocsr().astype(np.float32); L = np.maximum(np.asarray(Y.sum(1)).ravel(), 1)
    Lg = (sp.diags(1e4 / L) @ Y).tocsr(); Lg.data = np.log1p(Lg.data)
    return A.obs, np.asarray(A.obsm['spatial'], float), Y, L, np.asarray(Lg.mean(1)).ravel()      # last: per-spot mean over genes of log


def by_region(obs, Y, L, mlog):
    reg = obs.region.astype(str).values; lay = obs.layer.astype(str).values
    ok = np.isin(lay, LAY) & ~np.isin(reg, BAD); rc = pd.Series(reg[ok]).value_counts(); good = rc[rc >= 30].index
    rows = []
    for r in good:
        i = np.flatnonzero(ok & (reg == r))
        pooled = np.log1p(1e4 * np.asarray(Y[i].sum(0)).ravel() / L[i].sum()).mean()
        rows.append(dict(region=r, umi=L[i].mean(), meanlog=mlog[i].mean(), pooled=pooled, n=len(i)))
    D = pd.DataFrame(rows); D['meanlog'] -= D.meanlog.mean(); D['pooled'] -= D.pooled.mean(); return D


def style(ax):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.titlecolor': INK})
fig = plt.figure(figsize=(15, 10.5), facecolor=SURF)
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.38, wspace=0.6)
data = {}
for k, (ds, title) in enumerate([('pfcserLSimR', 'Simulation SimR: random counts, real UMI per spot\n(truth: identical expression in every region)'),
                                 ('pfcserL', 'Real series T402–T428')]):
    obs, xy, Y, L, mlog = load(ds); data[ds] = (obs, xy, Y, L, mlog); D = by_region(obs, Y, L, mlog)
    ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3]); style(ax)
    for col, c, lab in (('meanlog', ORANGE, 'mean of logs  (as now)'), ('pooled', BLUE, 'log of sums')):
        ax.scatter(D.umi, D[col], s=46, color=c, edgecolor=SURF, linewidth=1.5, zorder=3)
        b = np.polyfit(np.log(D.umi), D[col], 1); xs = np.linspace(D.umi.min(), D.umi.max(), 50)
        ax.plot(xs, np.polyval(b, np.log(xs)), color=c, lw=2, zorder=2)
        r = np.corrcoef(np.log(D.umi), D[col])[0, 1]
        ax.annotate(f'{lab}\nr = {r:+.2f}, range {D[col].max() - D[col].min():.3f}', xy=(xs[-1], np.polyval(b, np.log(xs[-1]))),
                    xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color=INK)
    ax.axhline(0, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_xscale('log'); ax.set_xlabel('mean UMI per spot in the region (log scale)'); ax.set_ylabel('mean over genes of log1p(CP10k)\n(centred over regions)')
    ax.set_title(title, fontsize=11, loc='left'); ax.set_xlim(D.umi.min() * 0.95, D.umi.max() * 1.35)
    print(ds, D[['meanlog', 'pooled']].agg(['min', 'max']).round(4).to_dict())

# ---- row 2: one SimR section on the tissue
obs, xy, Y, L, mlog = data['pfcserLSimR']; sec = obs.slice.astype(str).values; m = np.flatnonzero(sec == SEC)
G = grid_positions(np.array([SEC] * len(m)), xy[m])
Ld, _ = disk_mean(lambda idx: L[m][idx][:, None], 1, G, 1.5, chunk=1)
Ml, _ = disk_mean(lambda idx: mlog[m][idx][:, None], 1, G, 1.5, chunk=1)
Yc, _ = disk_mean(lambda idx: Y[m][idx].toarray(), Y.shape[1], G, 1.5, chunk=400)
Pl = np.log1p(1e4 * Yc / Ld).mean(1)
maps = [('UMI per spot (disk r 1.5)', Ld[:, 0], SEQ, None), ('mean of logs (as now)\ndeviation from section mean', Ml[:, 0] - Ml[:, 0].mean(), DIV, 'd'),
        ('log of sums\ndeviation from section mean', Pl - Pl.mean(), DIV, 'd')]
lim = max(np.percentile(np.abs(v), 99) for _, v, _, t in maps if t)
for j, (t, v, cm, kind) in enumerate(maps):
    ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2]); ax.set_facecolor(SURF)
    kw = dict(vmin=-lim, vmax=lim) if kind else dict(vmin=np.percentile(v, 1), vmax=np.percentile(v, 99))
    sc = ax.scatter(xy[m, 0], xy[m, 1], c=v, s=2.2, cmap=cm, marker='s', linewidths=0, **kw)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off'); ax.set_title(t, fontsize=10.5, loc='left')
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8, colors=INK2)
    if kind: print(t.split('\n')[0], 'corr with UMI map:', round(np.corrcoef(v, Ld[:, 0])[0, 1], 2))
fig.text(0.012, 0.465, f'SimR, section {SEC}: where the truth is flat, any pattern is an artefact', fontsize=11, color=INK, weight='bold')
fig.text(0.012, 0.975, 'Mean expression over all genes: mean of logs vs log of sums (both divided by the spot\'s UMI)', fontsize=13, color=INK, weight='bold')
fig.savefig(f'{OUT}/meanlog_vs_pooled.png', dpi=150, facecolor=SURF, bbox_inches='tight'); print('saved')
