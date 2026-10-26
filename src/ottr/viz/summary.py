"""Summary figures (step 10): model ranking, smoothing radius, training losses, reconstruction R2, axes vs atlas frontal PCs."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..config import cfg
from .style import save, MODEL_COL

METRIC_LABEL = {'mixing': 'section mixing', 'region_in_layer': 'region NMI in section', 'area_transfer': 'area transfer', 'layer': 'layer transfer', 'SA': 'SA axis', 'section_nmi': 'section NMI (lower = better)'}


def ranking(rank_csv=None, name='metrics_rank'):
    """Heat table of the ranking metrics (min-max shaded per column), models ordered by score."""
    T = pd.read_csv(rank_csv or cfg().tables('series_rank.csv'), index_col=0); cols = ['mixing', 'region_in_layer', 'area_transfer', 'layer', 'SA', 'section_nmi', 'score']
    V = T[cols]; Z = (V - V.min()) / (V.max() - V.min()).replace(0, 1); Z['section_nmi'] = 1 - Z['section_nmi']
    fig, ax = plt.subplots(figsize=(14, 0.62 * len(T) + 1.8), constrained_layout=True)
    ax.imshow(Z.values, cmap='Blues', vmin=-0.15, vmax=1.2, aspect='auto')
    for (i, j), v in np.ndenumerate(V.values): ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=12, color='white' if Z.values[i, j] > 0.65 else '#14213d')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([METRIC_LABEL.get(c, c) for c in cols], fontsize=12, rotation=15)
    ax.set_yticks(range(len(T))); ax.set_yticklabels([f'{int(p)}. {m}' for m, p in zip(T.index, T.place)], fontsize=13)
    ax.set_title('Serial series T402–T428: metrics and rank (score = mean of min-max scaled mixing, region, area, layer, SA)', fontsize=14, fontweight='bold', loc='left')
    return save(fig, name)


def smoothing_radius(csv=None, name='smoothing_radius'):
    T = pd.read_csv(csv or cfg().tables(f"smoothing_radius_{cfg().series['name']}.csv")); M = ['layer', 'mixing', 'region_in_layer', 'area_transfer', 'SA']
    fig, ax = plt.subplots(1, len(M), figsize=(5 * len(M), 4.2), constrained_layout=True); r0 = T[T.chosen].r.iloc[0]
    for a, m in zip(ax, M):
        a.plot(T.r, T[m], marker='o', color='#3a3936', lw=2); a.axvline(r0, color='#2a78d6', ls='--', lw=1.5)
        a.set_title(METRIC_LABEL.get(m, m), fontsize=13, fontweight='bold'); a.set_xlabel('disk radius, grid steps (1 step = 50 µm)'); a.grid(alpha=0.3)
    fig.suptitle(f'Smoothed PCA: metrics along the smoothing radius; chosen r = {r0:g} (smallest radius with the full layer transfer)', fontsize=14, fontweight='bold')
    return save(fig, name)


def losses(models, name='training_losses'):
    """models: [(key, label, loss_log.csv)]. Count NLL, feature reconstruction, FGW and total loss vs step."""
    PAN = [('recon_loss', 'Count reconstruction (negative log-likelihood)'), ('fe_loss', 'Feature reconstruction (L2 of the LGCN features)'),
           ('fgw_loss', 'FGW anchor loss'), ('loss_total', 'Total loss')]
    D = {k: pd.read_csv(f) for k, _, f in models if os.path.exists(f)}
    fig, ax = plt.subplots(1, 4, figsize=(27, 6.2), constrained_layout=True)
    for a, (col, ttl) in zip(ax, PAN):
        lo, hi = [], []
        for k, lbl, _ in models:
            if k not in D: continue
            d = D[k]; y = d[col].rolling(5, min_periods=1).mean(); a.plot(d.step, y, color=MODEL_COL.get(k, '#333333'), lw=2, label=lbl)
            a.annotate(f'{y.iloc[-1]:,.1f}' if col == 'fgw_loss' else f'{y.iloc[-1]:,.0f}', (d.step.iloc[-1], y.iloc[-1]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=10.5)
            late = d[d.step >= 1000][col]; lo.append(late.min()); hi.append(late.max())
        if lo: a.set_ylim(max(0, min(lo) * 0.9) if col != 'fgw_loss' else 0, max(hi) * 1.05)
        a.set_title(ttl, fontsize=13, fontweight='bold', loc='left'); a.set_xlabel('training step'); a.grid(alpha=0.25)
        for s in ('top', 'right'): a.spines[s].set_visible(False)
    ax[0].legend(fontsize=11, frameon=False)
    fig.suptitle('Training losses, series T402–T428 (20 000 steps)', fontsize=15, fontweight='bold')
    return save(fig, name, dpi=75)


def recon(csv=None, name='recon_r2'):
    T = pd.read_csv(csv or cfg().tables('recon_r2.csv'))
    MET = [('counts_spot_pooled_R2', 'counts · spot (CP10k)'), ('counts_pseudobulk_pooled_R2', 'counts · pseudobulk\n(section × region × layer)'),
           ('features_genes_pooled_R2', 'features · genes'), ('features_axf_pooled_R2', 'features · neighbourhood (axf)')]
    fig, ax = plt.subplots(figsize=(13, 5.6), constrained_layout=True); w = 0.8 / len(T); x = np.arange(len(MET)); cols = list(MODEL_COL.values())
    for i, r in enumerate(T.itertuples()):
        v = [getattr(r, m) for m, _ in MET]; b = ax.bar(x + (i - (len(T) - 1) / 2) * w, v, w * 0.92, color=MODEL_COL.get(r.model, cols[i % len(cols)]), label=r.model)
        for b_, val in zip(b, v): ax.text(b_.get_x() + b_.get_width() / 2, (val if np.isfinite(val) else 0) + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([t for _, t in MET], fontsize=12); ax.set_ylabel('pooled R² (1 − ΣSSE / ΣSST)'); ax.set_ylim(0, 1.05)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc='upper left'); ax.set_title('Reconstruction R² of the trained models', fontsize=14, fontweight='bold', loc='left')
    return save(fig, name, dpi=75)


def atlas_heatmap(csv=None, name='axes_vs_atlas'):
    S = pd.read_csv(csv or cfg().tables('atlas/axes_vs_frontal_models.csv')); models = list(dict.fromkeys(S.model)); kinds = ['pseudobulk areal', 'column PCA']
    fig, ax = plt.subplots(len(kinds), len(models), figsize=(6.4 * len(models), 4.2 * len(kinds)), constrained_layout=True, squeeze=False)
    for j, m in enumerate(models):
        for i, kl in enumerate(kinds):
            a = ax[i, j]; s = S[(S.model == m) & (S.analysis == kl)]; H = s[[f'rho_fPC{q}' for q in range(1, 11)]].values
            im = a.imshow(H, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            for (y, x_), v in np.ndenumerate(H):
                if abs(v) >= 0.4: a.text(x_, y, f'{v:.2f}', ha='center', va='center', fontsize=9, color='white' if abs(v) > 0.65 else '#14213d')
            a.set_xticks(range(10)); a.set_xticklabels([f'fPC{q}' for q in range(1, 11)], fontsize=9)
            a.set_yticks(range(len(s))); a.set_yticklabels([f'{r.comp}: {str(r.reading)[:30]}' for r in s.itertuples()], fontsize=9)
            if i == 0: a.set_title(m, fontsize=14, fontweight='bold')
            if j == 0: a.set_ylabel(kl, fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.5, label='Spearman ρ over atlas segments T408–T428')
    fig.suptitle('Our axes vs the frontal PCA of the atlas (fPC2 = SA; fPC4, fPC9 = medial ↔ orbital; fPC10 = arch)', fontsize=15, fontweight='bold')
    return save(fig, name, dpi=65)
