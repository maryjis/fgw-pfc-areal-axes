"""Inputs of the adult serial series (step 01).  python -m ottr.data.series [--ds pfcserL] [--sections T402,...]

Sections have manual layers (obs.layer) and atlas regions (obs.region) but no columns. Writes, in the working directory:
  runs/input/<ds>/<T>_adult_Sub-CTX_b100_filtered.h5ad   symlinks to the raw files (the training input of INSPIRE)
  inputs/<ds>.h5ad          raw counts of the SVG genes (QC and spot order exactly as the INSPIRE input), obs: slice, layer,
                            region, library_size (all genes), col / arc / depth (Laplace-streamline columns), spot_id
  inputs/<ds>_columns.csv   one row per column: majority fine region (area_level3), coarse area (area2), centroid,
                            SA annotation of the coarse area and the Ridge SA index of the frozen adult SA model."""
import os, argparse
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from ..config import cfg, ensure
from ..utils.columns import build_laplace, build_nearest
from ..utils.labels import AREA, SA_ANNO
from .sa_index import sa_index, lognorm_blocks, LAYERS


def qc(path):
    """QC exactly as the INSPIRE input (ottr.inspire.inputs.load_sections)."""
    a = sc.read_h5ad(path); a.var_names_make_unique()
    keep = ~pd.Index(a.var_names).isna() & ~a.var_names.str.upper().str.startswith('MT-')
    a = a[:, keep].copy(); sc.pp.filter_cells(a, min_genes=200); sc.pp.filter_genes(a, min_cells=3)
    return a


def prepare(ds, sections, laplace=True):
    c = cfg(); ind = ensure(c.run_input(ds)); ensure(os.path.dirname(c.inputs(ds)))
    for t in sections:
        l = f'{ind}/{t}_adult_Sub-CTX_b100_filtered.h5ad'
        if not os.path.lexists(l): os.symlink(f"{c.path('raw_adult')}/{t}_b100_anndata.h5ad", l)
    files = sorted(f'{ind}/{f}' for f in os.listdir(ind) if f.endswith('_b100_filtered.h5ad'))
    ads, shared = [], None
    for p in files:
        a = qc(p); ads.append(a); shared = set(a.var_names) if shared is None else shared & set(a.var_names)
    svg = [g for g in open(c.path('svg_genes')).read().split() if g in shared]
    build = build_laplace if laplace else build_nearest
    obs, Xs, xys, cols_of, raw_for_sa = [], [], [], {}, {}
    for i, (p, a) in enumerate(zip(files, ads)):
        sid = os.path.basename(p).replace('_b100_filtered.h5ad', ''); t = sid.split('_')[0]
        lib = np.asarray(a.X.sum(1)).ravel().astype(np.float32); lib[lib <= 0] = 1
        xy = np.asarray(a.obsm['spatial'], float); lay = a.obs.layer.astype(str).values
        reg = a.obs.region.astype(str).replace({'None': 'nan'}).values
        try: col, s, d = build(xy, lay)
        except Exception as e:                  # pole-tip sections with only L1-L2: no columns
            print(sid, 'no columns:', repr(e)[:120], flush=True); col = np.full(len(lay), -1); s = d = np.full(len(lay), np.nan)
        Xs.append(sp.csr_matrix(a[:, svg].X).astype(np.float32)); xys.append(xy)
        obs.append(pd.DataFrame({'spot_id': [f'{b}-{i}' for b in a.obs_names.astype(str)], 'slice': t, 'slice_full': sid, 'group': 'PFC',
                                 'animal': t, 'region': reg, 'layer': lay, 'library_size': lib, 'col': col, 'arc': s, 'depth': d}))
        fn = lognorm_blocks(sp.csr_matrix(a.X), lib, 'pseudobulk')
        for cc in np.unique(col[col >= 0]):
            m = col == cc; cols_of[(t, int(cc))] = {l: np.flatnonzero(m & (lay == l)) for l in LAYERS}
        raw_for_sa[t] = (fn, list(a.var_names))
        print(sid, a.n_obs, 'columns', int(col.max()), 'spots in columns', int((col >= 0).sum()), flush=True)
    O = pd.concat(obs, ignore_index=True).set_index('spot_id', drop=False)
    A = ad.AnnData(sp.vstack(Xs).tocsr(), obs=O, var=pd.DataFrame(index=svg)); A.obsm['spatial'] = np.vstack(xys).astype(np.float32)
    A.write_h5ad(c.inputs(ds), compression='gzip')
    sai = {}
    for t, (fn, genes_all) in raw_for_sa.items():   # the gene index differs per section after QC
        co = {k: v for k, v in cols_of.items() if k[0] == t}
        if co: sai.update(sa_index(fn, co, genes_all))
    rows = []
    for (t, cc), bl in cols_of.items():
        m = (O.slice.values == t) & (O.col.values == cc); r = O.region.values[m]; r = r[r != 'nan']
        lev3 = pd.Series(r).value_counts().index[0] if len(r) else 'nan'
        rows.append(dict(slice=t, col=cc, area_level3=lev3, mean_x=O.loc[m].index.size and A.obsm['spatial'][m, 0].mean(), mean_y=A.obsm['spatial'][m, 1].mean(),
                         area2=AREA.get(lev3, 'nan'), SA=SA_ANNO.get(AREA.get(lev3, ''), 'unassigned'), Ridge_SA_index=sai.get((t, cc), np.nan)))
    C = pd.DataFrame(rows).sort_values(['slice', 'col']); C.to_csv(c.columns(ds), index=False)
    print(ds, 'spots', A.n_obs, 'genes', len(svg), 'columns', len(C))


if __name__ == '__main__':
    P = argparse.ArgumentParser(); P.add_argument('--ds', default=cfg().series['name']); P.add_argument('--sections', default=None)
    a = P.parse_args()
    prepare(a.ds, a.sections.split(',') if a.sections else cfg().series['sections'], laplace=a.ds.endswith('L'))
