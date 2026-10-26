"""Shared pieces of the FGW anchor construction (no model training): the dataset with its columns and depth, the section
pairs to align, the representation (anchor space) of two sections, column blocks and column profiles.
Labels (region, area) are NOT used to build anchors - only to report how good the couplings are."""
import os
import numpy as np
import pandas as pd
import anndata as ad
from ..config import cfg
from ..utils.columns import layer_depth

LAYERS = ['L2', 'L3', 'L4', 'L5', 'L6']
NEED = ['L2', 'L3', 'L5', 'L6']                     # layers a column must contain (>= 3 spots each)


class Data:
    """One dataset of the working directory (inputs/<ds>.h5ad + inputs/<ds>_columns.csv)."""

    def __init__(self, ds, pairs_k=2):
        c = cfg(); self.ds = ds; self.pairs_k = pairs_k
        A = ad.read_h5ad(c.inputs(ds), backed='r'); o = A.obs.copy()
        self.obs = o; self.xy = np.asarray(A.obsm['spatial'], float)
        self.sl = o.slice.astype(str).values; self.grp = o.group.astype(str).values
        self.layer = o.layer.astype(str).values; self.col = o['col'].values.astype(int); self.region = o.region.astype(str).values
        # serial series: L6 optional (the frontal pole tip has none)
        self.need = ['L2', 'L3', 'L5'] if ds.startswith(('pfc405', 'pfcser')) else NEED
        self.slices = list(dict.fromkeys(self.sl))
        C = pd.read_csv(c.columns(ds)); C['area2'] = C.area2.astype(str); self.columns = C
        self.depth = layer_depth(self.sl, self.layer, self.xy)
        amap = C.set_index(['slice', 'col']).area2.to_dict()          # spot -> its column's coarse area (for reports only)
        self.area2 = np.array([amap.get((s, cc), 'nan') if cc >= 0 else 'nan' for s, cc in zip(self.sl, self.col)], dtype=object)

    def pairs(self):
        """Section pairs to align. Serial series: sections with columns in section order, neighbours up to pairs_k apart
        (k = 2: +-1 and +-2). Other datasets: all pairs within a group."""
        if self.ds.startswith('pfcser'):
            ss = sorted([s for s in self.slices if ((self.sl == s) & (self.col >= 0)).sum() > 0], key=lambda t: int(t[1:]))
            return [(ss[i], ss[i + d]) for d in range(1, self.pairs_k + 1) for i in range(len(ss) - d)]
        out = []
        for g in dict.fromkeys(self.grp):
            ss = [s for s in self.slices if self.grp[self.sl == s][0] == g]
            out += [(ss[i], ss[j]) for i in range(len(ss)) for j in range(i + 1, len(ss))]
        return out

    def space(self, emb, idx_a, idx_b):
        """Anchor space of the spots of two sections: embedding emb, z-scored over the union of both sections."""
        Z = np.load(cfg().emb(self.ds, emb)).astype(np.float32)
        U = np.concatenate([Z[idx_a], Z[idx_b]]); mu, sd = U.mean(0), U.std(0) + 1e-9
        return [(Z[idx_a] - mu) / sd, (Z[idx_b] - mu) / sd]


def default_space(ds):
    """Anchor space: disk-smoothed PCA r 1.5 (centred) for the adult PFC; the gene-SD smoothed PCA for the young brains."""
    return 'curve_r1.5_centred' if ds.startswith('pfc') else 'smoothed_pca32_centred'


def column_blocks(D, s):
    """Columns of section s that contain every required layer with >= 3 spots (L4 optional). Returns column ids (ordered along
    the cortex), per column {layer: global spot indices}, the arc-length coordinate of the column centroids and the areas."""
    m = np.flatnonzero((D.sl == s) & (D.col >= 0))
    df = pd.DataFrame({'col': D.col[m], 'layer': D.layer[m], 'i': m})
    cols, blocks = [], []
    for cc, g in df.groupby('col'):
        bl = {l: g.i.values[g.layer.values == l] for l in LAYERS}
        if any(len(bl[l]) < 3 for l in D.need): continue
        cols.append(int(cc)); blocks.append({l: v for l, v in bl.items() if len(v) >= 3})
    C = D.columns[D.columns.slice == s].set_index('col').loc[cols]
    xy = C[['mean_x', 'mean_y']].values; arc = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    return np.array(cols), blocks, arc, C.area2.values


def profiles(Z, blocks, pos):
    """Column profiles: equal-weight mean over layers of the layer means of Z (rows via pos(global idx)), z-scored within the section."""
    P = np.array([np.mean([Z[pos(b)].mean(0) for b in bl.values()], axis=0) for bl in blocks])
    return (P - P.mean(0)) / (P.std(0) + 1e-9)


def local_index(ia):
    """Vectorised map global spot index -> position within ia."""
    return np.vectorize({g: i for i, g in enumerate(ia)}.get, otypes=[np.int64])


def is_ring(D, s, cols, arc):
    """A section is a ring if the gap between its end columns is < 15% of the ribbon length. Returns (ring?, ring length)."""
    xy = D.columns[D.columns.slice.astype(str) == s].set_index('col').loc[cols, ['mean_x', 'mean_y']].values
    gap = np.linalg.norm(xy[0] - xy[-1]); L = arc[-1] - arc[0] + gap
    return gap < 0.15 * L, L
