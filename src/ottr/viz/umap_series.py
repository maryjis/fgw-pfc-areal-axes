"""UMAP of the series latents (step 10): rows = section number, region (label at the density peak), layer; one column per model.
UMAP settings as the INSPIRE figures: 30 neighbours, min_dist 0.3, correlation metric, random_state 0; coordinates cached in
work/umap/<ds>__<embedding>.npy.  plot(models) with models = [(embedding, label)]."""
import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
from ..config import cfg, ensure
from .style import LAYER_COL, save


def umap_coords(ds, emb):
    c = cfg(); f = os.path.join(ensure(os.path.join(c.work, 'umap')), f'{ds}__{emb}.npy')
    if not os.path.exists(f):
        import umap
        Z = np.load(c.emb(ds, emb)).astype(np.float32); Z = Z[:, Z.std(0) > 0]
        np.save(f, umap.UMAP(n_neighbors=30, min_dist=0.3, metric='correlation', random_state=0).fit_transform(Z).astype(np.float32))
    return np.load(f)


def plot(models, name='umap_series', ds=None):
    c = cfg(); ds = ds or c.series['name']
    o = ad.read_h5ad(c.inputs(ds), backed='r').obs; sec = o.slice.astype(str).values; lay = o.layer.astype(str).values; reg = o.region.astype(str).values
    num = np.array([int(s[1:]) for s in sec]); RG = ~np.isin(reg, ['nan', 'None', 'remove', 'unassigned']); regs = list(pd.Series(reg[RG]).value_counts().index)
    PAL = list(plt.cm.tab20(np.arange(20))) + list(plt.cm.tab20b(np.arange(0, 20, 2))) + list(plt.cm.Dark2(np.arange(8)))
    RP = {r: PAL[i % len(PAL)] for i, r in enumerate(regs)}; ST = [pe.withStroke(linewidth=3, foreground='white')]
    fig, ax = plt.subplots(3, len(models), figsize=(6.4 * len(models), 12.6), constrained_layout=True, squeeze=False); od = np.random.default_rng(0).permutation(len(o))
    for j, (e, n) in enumerate(models):
        U = umap_coords(ds, e)
        ax[0, j].scatter(U[od, 0], U[od, 1], s=0.3, c=num[od], cmap='viridis', linewidths=0, rasterized=True)
        a = ax[1, j]; a.scatter(U[od, 0], U[od, 1], s=0.3, c=[RP.get(x, (0.9, 0.89, 0.87, 1)) for x in reg[od]], linewidths=0, rasterized=True)
        lo, hi = U.min(0), U.max(0)
        for r_ in regs:
            P = U[reg == r_]
            if len(P) < 1500: continue
            H, xe, ye = np.histogram2d(P[:, 0], P[:, 1], bins=60, range=[[lo[0], hi[0]], [lo[1], hi[1]]]); ix, iy = np.unravel_index(gaussian_filter(H, 1.5).argmax(), H.shape)
            a.text((xe[ix] + xe[ix + 1]) / 2, (ye[iy] + ye[iy + 1]) / 2, r_, fontsize=11, fontweight='bold', color=RP[r_], ha='center', va='center', path_effects=ST)
        ax[2, j].scatter(U[od, 0], U[od, 1], s=0.3, c=[LAYER_COL.get(x, '#dcdad4') for x in lay[od]], linewidths=0, rasterized=True)
        ax[0, j].set_title(n, fontsize=19, fontweight='bold')
        for i, t in enumerate(('section (T402 → T428)', 'region', 'layer')):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([]); [v.set_visible(False) for v in ax[i, j].spines.values()]
            if j == 0: ax[i, j].set_ylabel(t, fontsize=17)
    fig.legend(handles=[plt.Line2D([], [], marker='o', ls='', ms=10, color=LAYER_COL[l], label=l) for l in LAYER_COL], loc='outside lower center', ncol=6, fontsize=16, frameon=False)
    return save(fig, name, dpi=55)
