"""Axes of one model on the tissue and regions on the axes (step 10).

tissue(model)   components 1-5 of the spot PCA / column PCA / areal model / spot-level model on five sections side by side
                (red high, blue low), last panel = regions; title colour = reading (SA blue, areal purple, section effect orange)
regions(model)  region means on component 1 against components 2-5 (spot / column / areal analysis); component 1 signed to
                rise with the region SA index; colours = coarse area
`model` = (multi_axes tag, label inside that run, spot_axes tag, display name, file key)."""
import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from ..config import cfg
from ..areal.multi_axes import axes_dir
from ..utils.labels import AREA
from .style import AREA_COL, reading_colour, save

SECS = ['T407', 'T413', 'T419', 'T423', 'T428']; NAME = {'spot': 'spot PC', 'column': 'column PC', 'pseudobulk': 'areal axis', 'spotmodel': 'spot-level axis'}


def _load(tag, label):
    c = cfg(); ds = c.series['name']
    T = pd.read_csv(c.tables(f'axes_{ds}{tag}_comp.csv')); T = T[T.model == label]
    Z = np.load(os.path.join(axes_dir(), f'{ds}{tag}_{label.replace(" ", "_")}.npz'), allow_pickle=True)
    return T, Z


def tissue(tag, label, spot_tag, title, key):
    c = cfg(); ds = c.series['name']; A = ad.read_h5ad(c.inputs(ds), backed='r'); o = A.obs; xy = np.asarray(A.obsm['spatial'], float)
    sec = o.slice.astype(str).values; lay = o.layer.astype(str).values; reg = o.region.astype(str).values; col = o.col.values.astype(int); u = o.depth.values.astype(float)
    T, Z = _load(tag, label)
    SA = pd.read_csv(c.tables(f'spot_axes_{ds}{spot_tag}.csv')); SA = SA[(SA.variant == 'layer+depth') & (SA.model == label)].set_index('component')
    okm = np.isin(lay, ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']) & np.isfinite(u); SM = np.full((len(o), 5), np.nan)
    SM[okm] = np.load(os.path.join(axes_dir(), f'spot_axes_scores{spot_tag}_layer+depth_{label.replace(" ", "_")}.npy'))
    SH, x0 = {}, 0.0
    for s in SECS:
        p = xy[sec == s]; SH[s] = (p.min(0), x0); x0 += np.ptp(p[:, 0]) + 500

    def draw(a, v, ttl, colr, cat=False):
        for s in SECS:
            q = sec == s; mn, off = SH[s]; p = xy[q] - mn; vv = v[q]
            if cat: a.scatter(p[:, 0] + off, -p[:, 1], s=0.5, c=vv, linewidths=0, rasterized=True); continue
            ok = np.isfinite(vv); lim = np.nanpercentile(np.abs(v[np.isin(sec, SECS)]), 98)
            a.scatter(p[:, 0] + off, -p[:, 1], s=0.5, c='#ecebe6', linewidths=0, rasterized=True)
            a.scatter(p[ok, 0] + off, -p[ok, 1], s=0.5, c=vv[ok], cmap='RdBu_r', vmin=-lim, vmax=lim, linewidths=0, rasterized=True)
        a.set_title(ttl, fontsize=15, fontweight='bold', color=colr, linespacing=1.3); a.set_aspect('equal'); a.axis('off')

    f = lambda x: '–' if pd.isna(x) else f'{x:.2f}'
    RG = sorted(set(reg) - {'nan', 'None', 'remove', 'unassigned'}); cm = plt.get_cmap('tab20', len(RG)); rc = {r: cm(i) for i, r in enumerate(RG)}
    rcol = np.array([rc.get(r, (0.9, 0.9, 0.9, 1)) for r in reg]); out = []
    for an, ttl in [('spot', 'spot PCA of the latent'), ('column', 'column PCA'), ('pseudobulk', 'areal-axis model (pseudobulk)'), ('spotmodel', 'spot-level areal model')]:
        fig, ax = plt.subplots(2, 3, figsize=(30, 11.5), constrained_layout=True)
        for j in range(5):
            if an == 'spotmodel':
                r = SA.loc[j + 1]; v = SM[:, j]
                rd = 'SA axis' if (r.SA_r_region_means >= 0.6 and r.region_eta2_in_layer >= 0.3) else 'region axis (not SA)' if r.region_eta2_in_layer >= 0.3 else 'within-layer variation'
                st = f'region η² {f(r.region_eta2_in_layer)} · SA r {f(r.SA_r_region_means)} · stability {f(r.axis_stability_odd_even)}'
            else:
                r = T[(T.analysis == an) & (T.comp == j + 1)].iloc[0]; rd = r.reading
                if an == 'spot': v = Z['spot_scores'][:, j].astype(float); st = f'layer η² {f(r.layer_eta2)} · region η² {f(r.region_eta2)} · SA ρ {f(r.SA_rho)}'
                elif an == 'column':
                    d = dict(zip(Z['col_keys'], Z['col_scores'][:, j])); v = np.array([d.get(f'{s}|{cc}', np.nan) for s, cc in zip(sec, col)], float)
                    st = f'SA ρ {f(r.SA_rho)} · area η² {f(r.region_eta2)} · section η² {f(r.section_eta2)}'
                else:
                    d = dict(zip(Z['pb_regions'], Z['pb_mu'][:, j])); v = np.array([d.get(x, np.nan) for x in reg], float); st = f'split-half {f(r.splithalf)} · SA r {f(r.SA_r)}'
            draw(ax.flat[j], v, f'{NAME[an]} {j + 1} — {r.var_share * 100:.0f}%: {rd}\n{st}', reading_colour(rd))
        draw(ax.flat[5], rcol, 'regions (reference)', '#14213d', cat=True)
        fig.suptitle(f'{title} · {ttl} · sections ' + ', '.join(SECS), fontsize=17, fontweight='bold')
        out.append(save(fig, f'tissue_{key}_{an}', dpi=55))
    return out


def regions(tag, label, title, key):
    c = cfg(); ds = c.series['name']; o = ad.read_h5ad(c.inputs(ds), backed='r').obs
    reg = o.region.astype(str).values; sec = o.slice.astype(str).values; col = o.col.values.astype(int)
    C = pd.read_csv(c.columns(ds)); SA = C[C.Ridge_SA_index.notna()].groupby('area_level3').Ridge_SA_index.mean(); RG = np.isin(reg, list(AREA))
    T, Z = _load(tag, label); ST = [pe.withStroke(linewidth=2.5, foreground='white')]; out = []
    for an in ('spot', 'column', 'pseudobulk'):
        if an == 'pseudobulk': R = pd.DataFrame(Z['pb_mu'][:, :5], index=Z['pb_regions'])
        else:
            if an == 'spot': V = Z['spot_scores'][:, :5].astype(float)
            else:
                d = dict(zip(Z['col_keys'], Z['col_scores'][:, :5].astype(float))); V = np.array([d.get(f'{s}|{cc}', [np.nan] * 5) for s, cc in zip(sec, col)], float)
            R = pd.DataFrame(V[RG]).groupby(reg[RG]).mean()
        R = R.dropna(); R = R[R.index.isin(list(AREA))]
        if np.corrcoef(R[0], SA.reindex(R.index).fillna(SA.mean()))[0, 1] < 0: R[0] = -R[0]
        t = T[T.analysis == an].set_index('comp')
        fig, ax = plt.subplots(1, 4, figsize=(30, 7.2), constrained_layout=True)
        for j, k in enumerate(range(1, 5)):
            a = ax[j]
            for r, row in R.iterrows():
                cc = AREA_COL.get(AREA.get(r), '#bbbbbb'); a.scatter(row[0], row[k], s=70, color=cc, edgecolor='white', linewidth=0.6)
                a.text(row[0], row[k], r, fontsize=10.5, color=cc, ha='center', va='bottom', path_effects=ST)
            a.axhline(0, color='#cccccc', lw=0.8); a.axvline(0, color='#cccccc', lw=0.8)
            sh = lambda n: f' · split-half {t.loc[n].splithalf:.2f}' if an == 'pseudobulk' and pd.notna(t.loc[n].splithalf) else ''
            a.set_xlabel(f'{NAME[an]} 1 — {t.loc[1].var_share * 100:.0f}% · {t.loc[1].reading}{sh(1)}', fontsize=13, fontweight='bold')
            a.set_ylabel(f'{NAME[an]} {k + 1} — {t.loc[k + 1].var_share * 100:.0f}%\n{t.loc[k + 1].reading}{sh(k + 1)}', fontsize=13, fontweight='bold')
        ax[0].set_title(f"{title} · {NAME[an].replace('axis', 'axe')}s", fontsize=17, fontweight='bold', loc='left')
        fig.legend(handles=[plt.Line2D([], [], marker='o', ls='', ms=11, color=cc, label=a_) for a_, cc in AREA_COL.items()], loc='outside lower center', ncol=7, fontsize=15, frameon=False)
        out.append(save(fig, f'regions_{key}_{an}', dpi=55))
    return out
