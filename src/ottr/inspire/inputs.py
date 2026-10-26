"""Training inputs of INSPIRE from per-section raw bin100 files.

For every section (one h5ad of raw counts with obsm['spatial']):
  1. QC exactly as INSPIRE: drop unnamed and mitochondrial genes, cells with < 200 genes, genes in < 3 cells;
  2. genes = the fixed SVG list (data/adult5_svg_genes.txt) intersected with the genes present in every section
     (or exactly a forced list, missing genes as zero columns - used to encode the young brains with the adult model);
  3. normalisation (as INSPIRE utils.preprocess): CP10k over ALL genes of the section, then log1p ->
     lognorm = log1p(1e4 * x_g / sum_all_genes x);  raw counts of the model genes and their sum (library size) are kept
     for the count likelihood;
  4. spatial graph: radius graph with radius = 1.5 x the median nearest-neighbour distance (8 neighbours on the bin100
     grid), self loops, symmetric normalisation D^-1/2 (A + I) D^-1/2;
  5. node features = [lognorm, graph @ lognorm]  (the LGCN input of INSPIRE: own expression + smoothed expression);
  6. 8 nearest grid neighbours of every spot (for the patch-variance loss).
Returns one AnnData per section in the layout Model_LGCN expects (obsm node_features, count, spatial, neighbor_idx;
obs slice, slice_id, library_size) and the concatenated adata_full that receives the latent."""
import os, glob
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from scipy.spatial import cKDTree


def load_sections(slice_dir, glob_pattern='*_b100_filtered.h5ad', min_genes=200, min_cells=3):
    """Read and QC every section file in slice_dir (sorted by name). Returns (section ids, AnnData list, shared genes)."""
    paths = sorted(glob.glob(os.path.join(slice_dir, glob_pattern)))
    ids = [os.path.basename(p).replace('_b100_filtered.h5ad', '') for p in paths]
    ads, shared = [], None
    for p in paths:
        a = sc.read_h5ad(p); a.var_names_make_unique()
        keep = ~pd.Index(a.var_names).isna() & ~a.var_names.str.upper().str.startswith('MT-')
        a = a[:, keep].copy()
        sc.pp.filter_cells(a, min_genes=min_genes); sc.pp.filter_genes(a, min_cells=min_cells)
        a.obsm['spatial'] = np.asarray(a.obsm['spatial'], dtype=np.float32)
        ads.append(a); shared = set(a.var_names) if shared is None else shared & set(a.var_names)
        print(f'  loaded {os.path.basename(p)}: {a.shape}', flush=True)
    return ids, ads, sorted(shared)


def select_genes(ads, shared, svg_genes_file=None, force_genes_file=None):
    """Model genes: the SVG list restricted to the shared genes, or a forced list (zero columns added where missing)."""
    if force_genes_file:
        genes = [l.strip() for l in open(force_genes_file) if l.strip()]
        for i, a in enumerate(ads):
            miss = [g for g in genes if g not in set(a.var_names)]
            if miss:
                z = ad.AnnData(sp.csr_matrix((a.n_obs, len(miss)), dtype=np.float32), obs=a.obs.copy(), var=pd.DataFrame(index=miss))
                obsm = dict(a.obsm); a = ad.concat([a, z], axis=1, merge='first'); a.obsm['spatial'] = obsm['spatial']; ads[i] = a
            print(f'  forced genes: {len(genes)}, missing in section {i}: {len(miss)}')
        return genes
    genes = [l.strip() for l in open(svg_genes_file) if l.strip()]
    s = set(shared); genes = [g for g in genes if g in s]
    print(f'  {len(genes)} SVG genes present in all sections')
    return genes


def normalized_radius_graph(coords, rad_coef=1.5):
    """Radius graph (radius = rad_coef x median nearest-neighbour distance) with self loops, D^-1/2 A D^-1/2."""
    tree = cKDTree(coords)
    nn_dist, _ = tree.query(coords, k=min(2, coords.shape[0]))
    pos = nn_dist[:, 1] if nn_dist.ndim > 1 else nn_dist; pos = pos[pos > 0]
    radius = float(np.median(pos)) * rad_coef
    pairs = np.array(list(tree.query_pairs(radius)), dtype=np.int64); n = coords.shape[0]
    if pairs.size == 0:
        rows = cols = np.arange(n, dtype=np.int64)
    else:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n, dtype=np.int64)])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n, dtype=np.int64)])
    graph = sp.coo_matrix((np.ones(rows.shape[0], dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(graph.sum(axis=1)).reshape(-1); deg[deg <= 0] = 1.0
    inv_sqrt = np.power(deg, -0.5).astype(np.float32)
    graph = sp.diags(inv_sqrt) @ graph @ sp.diags(inv_sqrt)
    avg_neighbors = float(np.asarray((graph > 0).sum(axis=1)).reshape(-1).mean() - 1.0)
    return graph.tocsr(), radius, avg_neighbors


def build_inputs(ids, ads, genes, rad_coef=1.5, neighbor_k=8, min_concat_dist=500.0):
    """Per-section INSPIRE inputs (see module docstring) and the concatenated adata_full (coordinates shifted side by side)."""
    st_list = []
    for i, a in enumerate(ads):
        a.layers['counts'] = a.X.copy()
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)           # CP10k over all genes of the section, log1p
        raw = ad.AnnData(a.layers.pop('counts'), obs=a.obs, var=a.var)
        c = raw[:, genes].X.copy()
        counts = (c.toarray() if sp.issparse(c) else np.asarray(c)).astype(np.float32, copy=False)
        library_size = counts.sum(axis=1).astype(np.float32); library_size[library_size <= 0] = 1.0   # over the model genes
        lognorm = a[:, genes].X
        lognorm = (lognorm.toarray() if sp.issparse(lognorm) else np.asarray(lognorm)).astype(np.float32)
        graph, radius, avg_nb = normalized_radius_graph(a.obsm['spatial'], rad_coef)
        axf = (graph @ lognorm).astype(np.float32, copy=False)
        node_features = np.concatenate([lognorm, axf], axis=1).astype(np.float32, copy=False)
        _, nbr = cKDTree(a.obsm['spatial']).query(a.obsm['spatial'], k=neighbor_k + 1)
        st = ad.AnnData(np.zeros((a.n_obs, len(genes)), dtype=np.float32)); st.var_names = genes
        st.obs_names = [f'{x}-{i}' for x in a.obs_names.astype(str)]
        st.obs['slice'] = i; st.obs['slice'] = st.obs['slice'].astype(int); st.obs['slice_id'] = ids[i]
        st.obs['library_size'] = library_size
        st.obsm['count'] = counts; st.obsm['node_features'] = node_features
        st.obsm['spatial'] = a.obsm['spatial'].copy(); st.obsm['neighbor_idx'] = nbr[:, 1:].astype(np.int64)
        st_list.append(st)
        print(f'  {ids[i]}: n={a.n_obs}, radius={radius:.2f}, avg_neighbors={avg_nb:.2f}, node_features={node_features.shape}', flush=True)
    shifted, x_off = [], 0.0
    for st in st_list:
        xy = st.obsm['spatial'].copy(); xy[:, 0] = xy[:, 0] - xy[:, 0].min() + x_off; xy[:, 1] = xy[:, 1] - xy[:, 1].min()
        x_off = float(xy[:, 0].max() + min_concat_dist)
        t = ad.AnnData(np.zeros((st.n_obs, 1), dtype=np.float32), obs=st.obs.copy()); t.var_names = ['placeholder']; t.obsm['spatial'] = xy
        shifted.append(t)
    return st_list, ad.concat(shifted, join='outer')
