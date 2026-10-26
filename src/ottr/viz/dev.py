"""Development figures (step 10) for one AxesModel (ottr.areal.development).

spatial(M)       rows = adult sections T407 / T419 / T428 and every P0 / P32 / 3M / 7M section, columns = adult areal axes 1-5
                 (or column PCs) + areas; each age centred on its own mean, areal axes within each layer; one colour scale per axis
conservation(M)  per areal axis and age: share of the age's variance along the adult axis, best |cos| with the age's own PCs,
                 r of the 4 area means with adults, eta2 of the areas
age(M)           per areal axis one number per sample (median over its L1-L6 bins) against age, and the top-30 genes of each end"""
import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from ..config import cfg
from ..areal.development import LAYERS, AGES, col_profiles
from ..utils.labels import AREA as FINE2AREA, COARSE_DEV
from .style import DEV_COL, AGE_COL, COMP_COL, save

ADULT = ['T407', 'T419', 'T428']; NM = {'column': 'column PC', 'pseudobulk': 'areal axis'}


def spatial(M, title, analysis='pseudobulk'):
    c = cfg(); Z = M.npz; kf = Z['keepf']; T = M.comp

    def load(ds, emb):
        A = ad.read_h5ad(c.inputs(ds), backed='r'); o = A.obs
        return dict(E=np.load(c.emb(ds, emb)).astype(np.float64)[:, kf], sec=o.slice.astype(str).values, col=o.col.values.astype(int),
                    lay=o.layer.astype(str).values, reg=o.region.astype(str).values, xy=np.asarray(A.obsm['spatial'], float))
    AD = load(c.series['name'], M.adult_emb); m_lay = {l: AD['E'][AD['lay'] == l].mean(0) for l in LAYERS}
    _, aP, _ = col_profiles(AD['E'], AD['sec'], AD['col'], AD['lay']); m_col = aP.mean(0)

    def scores(D, keep):
        keys, P, _ = col_profiles(D['E'], D['sec'], D['col'], D['lay']); cs = (P - m_col) @ Z['col_V'][:5].T; d = {k: i for i, k in enumerate(keys)}
        ci = np.array([d.get((s, cc), -1) for s, cc in zip(D['sec'], D['col'])]); out = {'column': np.where(ci[:, None] >= 0, cs[np.maximum(ci, 0)], np.nan)}
        pb = np.full((len(D['E']), 5), np.nan)
        for j, l in enumerate(LAYERS):
            q = D['lay'] == l
            if q.any(): pb[q] = (D['E'][q] - m_lay[l]) @ Z['pb_G'][:5, j].T
        out['pseudobulk'] = pb
        for an in out:
            v = out[an]; v -= np.nanmean(v[keep], 0)
            if an == 'pseudobulk':
                for l in LAYERS:
                    q = D['lay'] == l; v[q] -= np.nanmean(v[q & keep], 0)
        return out
    AD['S'] = scores(AD, np.isin(AD['sec'], ADULT)); AD['area'] = np.array([COARSE_DEV.get(FINE2AREA.get(r, ''), 'nan') for r in AD['reg']])
    PANELS = [('adult', AD, s) for s in ADULT]
    for a in AGES[:-1]:
        D = load(f'dev{a}', M.dev_emb); D['S'] = scores(D, np.ones(len(D['E']), bool)); D['area'] = D['reg']; PANELS += [(a, D, s) for s in dict.fromkeys(D['sec'])]
    an = analysis; lim = {k: np.nanpercentile(np.abs(np.concatenate([D['S'][an][D['sec'] == s, k] for _, D, s in PANELS])), 98) for k in range(5)}
    fig, ax = plt.subplots(len(PANELS), 6, figsize=(6 * 3.6, 3.1 * len(PANELS) + 1.2), constrained_layout=True); t = T[T.analysis == an].set_index('comp')
    for i, (age, D, s) in enumerate(PANELS):
        q = D['sec'] == s; p = D['xy'][q] - D['xy'][q].min(0)
        for k in range(5):
            a = ax[i, k]; v = D['S'][an][q, k]; f = np.isfinite(v)
            a.scatter(p[:, 0], -p[:, 1], s=0.5, c='#ecebe6', linewidths=0, rasterized=True)
            a.scatter(p[f, 0], -p[f, 1], s=0.5, c=v[f], cmap='RdBu_r', vmin=-lim[k], vmax=lim[k], linewidths=0, rasterized=True); a.set_aspect('equal'); a.axis('off')
            if i == 0: a.set_title(f'adult {NM[an]} {k + 1}\n{t.loc[k + 1].reading}', fontsize=13, fontweight='bold')
            if k == 0: a.text(-0.06, 0.5, f'{age}\n{s}', transform=a.transAxes, rotation=90, ha='right', va='center', fontsize=13, fontweight='bold', color=AGE_COL[age])
        a = ax[i, 5]; a.scatter(p[:, 0], -p[:, 1], s=0.5, c=[DEV_COL.get(x, '#dddddd') for x in D['area'][q]], linewidths=0, rasterized=True); a.set_aspect('equal'); a.axis('off')
        if i == 0: a.set_title('areas', fontsize=13, fontweight='bold')
    fig.legend(handles=[plt.Line2D([], [], marker='o', ls='', ms=10, color=cc, label=n) for n, cc in DEV_COL.items()], loc='outside lower center', ncol=4, fontsize=14, frameon=False)
    fig.suptitle(f'Adult {title} {NM[an]}s 1-5 on adult and developing PFC sections (adult encoder; each age centred on its own mean'
                 + (', per layer' if an == 'pseudobulk' else '') + '; red high, blue low)', fontsize=16, fontweight='bold')
    return save(fig, f'dev_spatial_{M.key}_{an}', dpi=50)


def conservation(M, title):
    C = pd.read_csv(M.out('conservation.csv')); C = C[C.analysis == 'pseudobulk']; X_ = np.arange(len(AGES))
    MET = [('share', "share of the age's variance along the axis"), ('own_cos', "best |cos| with the age's own top-5 PCs"),
           ('r_area', 'r of the 4 area means with adults'), ('eta2_area', "η² of the 4 areas among the age's columns")]
    fig, ax = plt.subplots(1, 4, figsize=(26, 6.6), constrained_layout=True)
    for a, (m, lab) in zip(ax, MET):
        for k in range(5):
            s = C[C.comp == k + 1].set_index('age').reindex(AGES)
            a.plot(X_, s[m].values, marker='o', lw=2.6 if k in (0, 2) else 1.4, color=COMP_COL[k], label=f'{k + 1}: {s.reading.iloc[0]}')
        a.set_xticks(X_); a.set_xticklabels(AGES, fontsize=13); a.set_title(f'areal model · {lab}', fontsize=13, fontweight='bold'); a.grid(alpha=0.3)
        if m != 'share': a.set_ylim(-1.0 if m == 'r_area' else 0, 1.05)
    ax[0].legend(fontsize=11, frameon=False, loc='upper left')
    fig.suptitle(f'Adult {title} areal axes across development (latent from the adult encoder)', fontsize=16, fontweight='bold')
    return save(fig, f'dev_conservation_{M.key}', dpi=60)


def age(M, title, stat='median'):
    SUF = '' if stat == 'mean' else f'_{stat}'
    S = pd.read_csv(M.out(f'age_areal_samples{SUF}.csv')); RR = pd.read_csv(M.out(f'age_areal_rho{SUF}.csv')).set_index('what'); T = M.comp; T = T[T.analysis == 'pseudobulk'].set_index('comp')
    fig, ax = plt.subplots(2, 5, figsize=(31, 10.5), constrained_layout=True); X_ = np.arange(len(AGES)); rng = np.random.default_rng(0)
    for k in range(5):
        for r_, cols in enumerate([[f'axis{k + 1}'], [f'hi{k + 1}', f'lo{k + 1}']]):
            a = ax[r_, k]
            for col, cc in zip(cols, ['#14213d'] if r_ == 0 else ['#c2410c', '#1d4ed8']):
                for i, ag in enumerate(AGES):
                    v = S[S.age == ag][col].values; a.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=34, color=AGE_COL[ag] if r_ == 0 else cc, alpha=0.75, edgecolor='white', linewidth=0.5, zorder=3)
                    a.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=cc, lw=3, zorder=4)
                a.plot(X_, [S[S.age == ag][col].median() for ag in AGES], color=cc, lw=1.2, alpha=0.6)
            if r_ == 0: t = f'axis {k + 1}: {T.loc[k + 1].reading}\nρ(age) dev {RR.loc[f"axis {k + 1}"].rho_dev9:+.2f} · all {RR.loc[f"axis {k + 1}"].rho_all36:+.2f}'
            else: t = f'positive genes ρ {RR.loc[f"axis {k + 1} positive genes"].rho_dev9:+.2f} (red)\nnegative genes ρ {RR.loc[f"axis {k + 1} negative genes"].rho_dev9:+.2f} (blue)'
            a.set_title(t, fontsize=12.5, fontweight='bold'); a.set_xticks(X_); a.set_xticklabels(AGES, fontsize=12); a.grid(alpha=0.3); a.axhline(0, color='#bbbbbb', lw=0.8)
    fig.suptitle(f'Adult {title} areal axes across development: one number per sample ({stat} over its L1–L6 bins) and the top-30 genes of each end', fontsize=16, fontweight='bold')
    return save(fig, f'dev_age_{M.key}', dpi=60)
