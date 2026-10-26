"""Cortical columns of a section from its manual layers (no column annotation needed).

Two builders (both return per-spot column id (-1 = none), arc length along the cortex, and a depth):

build_nearest (centre line)
  1. relative depth d = d1 / (d1 + d6) from the distances to the nearest L1 and L6 spot;
  2. mid-depth centre line = L2-L6 spots with 0.42 <= d <= 0.58, joined in a radius graph (1.6 grid steps) whose edges may not
     cross holes or deep layers;
  3. per connected piece: double sweep (two Dijkstra runs) finds an end; the line is cut there (rings become open ribbons),
     a second double sweep finds the start; arc length s = Dijkstra distance from the start;
  4. every L2-L6 spot takes s of its nearest centre-line spot; columns = bins of s of width 415 units (~207 um at mid depth,
     reproducing the 94 reference columns of T419).

build_laplace (Laplace streamlines; used for the serial series 'pfcserL' and the young brains)
  1. depth u solves the Laplace equation on the 4-neighbour grid of the cortex with u = 0 on the pial edge of L1 and u = 1 on
     the edge of the deepest layer (no flux elsewhere); solid sections without a white-matter edge use the 20% of the
     deepest layer farthest from L1 as the u = 1 boundary;
  2. centre line at relative depth MIDD = 0.6 (|d - 0.6| <= 0.08), arc length as above;
  3. streamlines: every L2-L6 spot follows the steepest change of u towards the centre line (8-neighbour grid, <= 60 steps)
     and takes the arc length where it lands; columns = bins of width 405;
  4. a column lacking >= 3 spots of a required layer (narrow wedges at convex corners) joins its smaller neighbour.
Copied without changes in logic from the project scripts columns_build.py / columns_laplace.py."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra, connected_components

STEP = 100.0            # grid step of obsm['spatial'] (bin100)
WIDTH_NEAREST = 415.0   # column width of build_nearest
CUT = 700.0             # length cut off at the found end of a ring
WIDTH_LAPLACE = 405.0   # column width of build_laplace
MIDD = 0.6              # relative depth of the line along which build_laplace measures the arc length
LAY_ORDER = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
CORTEX = ['L2', 'L3', 'L4', 'L5', 'L6', 'L2/3/4']


def depth(xy, lay):
    """Relative depth d1 / (d1 + d6): 0 in L1, 1 in L6."""
    d1 = cKDTree(xy[lay == 'L1']).query(xy)[0]; d6 = cKDTree(xy[lay == 'L6']).query(xy)[0]
    v = d1 / (d1 + d6 + 1e-9); v[lay == 'L1'] = 0; v[lay == 'L6'] = 1
    return v


def _graph(P, xy_all=None, dep_all=None):
    """Radius graph (1.6 grid steps) of the centre-line spots P; with xy_all / dep_all an edge is kept only if its midpoint is
    itself a cortex spot at mid depth (0.3-0.7), so the line does not jump across holes or the deep layers."""
    t = cKDTree(P); pairs = t.query_pairs(1.6 * STEP, output_type='ndarray')
    if xy_all is not None:
        mid = (P[pairs[:, 0]] + P[pairs[:, 1]]) / 2
        dd, j = cKDTree(xy_all).query(mid)
        ok = (dd < 0.75 * STEP) & (dep_all[j] >= 0.3) & (dep_all[j] <= 0.7); pairs = pairs[ok]
    w = np.linalg.norm(P[pairs[:, 0]] - P[pairs[:, 1]], axis=1)
    return coo_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])), shape=(len(P), len(P))).tocsr()


def _far(G, start):
    """Farthest node from start (Dijkstra) and all distances (-1 = unreachable)."""
    d = dijkstra(G, indices=start); d[~np.isfinite(d)] = -1
    return int(np.argmax(d)), d


def _arc_of_pieces(G, P, width):
    """Arc length along every piece (>= 30 nodes) of the centre-line graph: cut at one end, double sweep for the start.
    Pieces are laid one after another (largest first) with a gap of 2 column widths."""
    ncomp, lab = connected_components(G, directed=False); sizes = np.bincount(lab); order = np.argsort(-sizes)
    s_mid = np.full(len(P), np.nan); offset = 0.0
    for c in order:
        if sizes[c] < 30: continue
        nodes = np.flatnonzero(lab == c); sub = G[nodes][:, nodes]
        e, _ = _far(sub, 0); e, _ = _far(sub, e)
        keep = np.linalg.norm(P[nodes] - P[nodes][e], axis=1) > CUT
        n2 = nodes[keep]; sub2 = G[n2][:, n2]; _, lab2 = connected_components(sub2, directed=False)
        n3 = n2[lab2 == np.argmax(np.bincount(lab2))]; sub3 = G[n3][:, n3]
        a, _ = _far(sub3, 0); a, _ = _far(sub3, a); _, dist = _far(sub3, a)
        s_mid[n3] = dist + offset; offset += dist.max() + 2 * width
    return s_mid


def build_nearest(xy, lay):
    lay = np.asarray(lay).astype(str); cortex = np.isin(lay, CORTEX); has6 = (lay == 'L6').any()
    if has6:
        d = depth(xy, lay); mid = cortex & (d >= 0.42) & (d <= 0.58)
    else:                                   # sections without L6 (frontal pole tip): centre line = L3
        d = np.full(len(xy), np.nan); mid = lay == 'L3'
    mi = np.flatnonzero(mid); P = xy[mi]
    G = _graph(P, xy[cortex], d[cortex]) if has6 else _graph(P)
    s_mid = _arc_of_pieces(G, P, WIDTH_NEAREST)
    ok = np.isfinite(s_mid)
    _, j = cKDTree(P[ok]).query(xy[cortex]); s = np.full(len(xy), np.nan); s[cortex] = s_mid[ok][j]
    col = np.full(len(xy), -1); col[cortex] = (s[cortex] // WIDTH_NEAREST).astype(int)
    u = np.unique(col[col >= 0]); remap = {c: k + 1 for k, c in enumerate(u)}
    return np.array([remap.get(c, -1) for c in col]), s, d


def laplace_depth(xy, lay):
    """Laplace depth u (0 at the pia, 1 at the white-matter edge) on the 4-neighbour grid. Returns u, grid index, key map."""
    ij = np.round((xy - xy.min(0)) / STEP).astype(int); key = {(a, b): k for k, (a, b) in enumerate(map(tuple, ij))}
    n = len(xy); rows, cols = [], []; edge = np.zeros(n, bool)
    for k, (a, b) in enumerate(map(tuple, ij)):
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = key.get((a + da, b + db))
            if j is None: edge[k] = True
            else: rows.append(k); cols.append(j)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    deepest = [l for l in LAY_ORDER if (lay == l).any()][-1]
    fixed0 = edge & (lay == 'L1'); fixed1 = edge & (lay == deepest)
    ncomp, lab = connected_components(A, directed=False)
    d1 = cKDTree(xy[lay == 'L1']).query(xy)[0]
    for c in range(ncomp):                  # solid pieces without a white-matter edge: deepest 20% of the deepest layer
        m = (lab == c) & (lay == deepest)
        if m.any() and not fixed1[lab == c].any(): fixed1 |= m & (d1 >= np.quantile(d1[m], 0.8))
    u = np.full(n, np.nan); fixed = fixed0 | fixed1; u[fixed0] = 0.0; u[fixed1] = 1.0
    for c in range(ncomp):                  # Dirichlet problem on every piece touching both boundaries
        m = lab == c
        if not (fixed0[m].any() and fixed1[m].any()): continue
        free = np.flatnonzero(m & ~fixed); fx = np.flatnonzero(m & fixed)
        deg = np.asarray(A[free].sum(1)).ravel()
        L = (coo_matrix((deg, (np.arange(len(free)), np.arange(len(free)))), shape=(len(free), len(free))) - A[free][:, free]).tocsc()
        u[free] = spsolve(L, A[free][:, fx] @ u[fx])
    return u, ij, key


def build_laplace(xy, lay, merge=True):
    lay = np.asarray(lay).astype(str); xy = np.asarray(xy, float)
    u, ij, key = laplace_depth(xy, lay)
    cortex = np.isin(lay, CORTEX) & np.isfinite(u)
    # the arc is measured at the layer mid-depth: in rings the u = 0.5 level of a Laplace field sits close to the short inner
    # boundary, which would compress the columns; u only sets the direction of the streamlines
    deepest = [l for l in LAY_ORDER if (lay == l).any()][-1]
    d1 = cKDTree(xy[lay == 'L1']).query(xy)[0]; dd = cKDTree(xy[lay == deepest]).query(xy)[0]
    dl = d1 / (d1 + dd + 1e-9); dl[lay == 'L1'] = 0; dl[lay == deepest] = 1
    mid = cortex & (np.abs(dl - MIDD) <= 0.08); mi = np.flatnonzero(mid); P = xy[mi]
    G = _graph(P, xy[cortex], dl[cortex])
    s_mid = _arc_of_pieces(G, P, WIDTH_LAPLACE)
    s_at = np.full(len(xy), np.nan); s_at[mi] = s_mid
    nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    nxt = np.full(len(xy), -1)
    for k in np.flatnonzero(cortex & ~mid):   # above the mid band: steepest increase of u; below: steepest decrease
        a, b = ij[k]; sign = 1.0 if dl[k] < MIDD else -1.0; best, bv = -1, 0.0
        for da, db in nb:
            j = key.get((a + da, b + db))
            if j is None or not np.isfinite(u[j]): continue
            v = sign * (u[j] - u[k]) / np.hypot(da, db)
            if v > bv + 1e-12: best, bv = j, v
        nxt[k] = best
    s = np.full(len(xy), np.nan)
    for k in np.flatnonzero(cortex):
        j, steps = k, 0
        while not mid[j] and nxt[j] >= 0 and steps < 60: j = nxt[j]; steps += 1
        if mid[j] and np.isfinite(s_at[j]): s[k] = s_at[j]
    col = np.full(len(xy), -1); ok = np.isfinite(s); col[ok] = (s[ok] // WIDTH_LAPLACE).astype(int)
    if merge:
        need = [l for l in ('L2', 'L3', 'L5', 'L6') if (lay == l).any()]
        for _ in range(3):
            ids = np.unique(col[col >= 0]); cnt = {c: {l: int(((col == c) & (lay == l)).sum()) for l in need} for c in ids}
            bad = [c for c in ids if min(cnt[c].values()) < 3]
            if not bad: break
            for c in bad:
                nbrs = [x for x in (c - 1, c + 1) if x in cnt]
                if not nbrs: continue
                tgt = min(nbrs, key=lambda x: sum(cnt[x].values())); col[col == c] = tgt
    u_ = np.unique(col[col >= 0]); remap = {c: i + 1 for i, c in enumerate(u_)}
    return np.array([remap.get(c, -1) for c in col]), s, u


def layer_depth(sl, layer, xy):
    """Relative cortical depth per section: d1 / (d1 + d_deep), d = distance to the nearest L1 / deepest-layer spot (L6, or
    L5 in sections with < 50 L6 spots). L1 -> 0, deepest layer -> 1, WM -> 1.1, unlabelled -> nan. (anchor_common.layer_depth)"""
    dep = np.full(len(sl), np.nan)
    for s in dict.fromkeys(sl):
        m = np.flatnonzero(sl == s); lay = layer[m]; p = xy[m]
        deep = 'L6' if (lay == 'L6').sum() >= 50 else 'L5'
        if not (lay == 'L1').any() or not (lay == deep).any(): continue
        d1 = cKDTree(p[lay == 'L1']).query(p)[0]; d6 = cKDTree(p[lay == deep]).query(p)[0]
        v = d1 / (d1 + d6 + 1e-9); v[lay == 'L1'] = 0; v[lay == deep] = 1; v[lay == 'WM'] = 1.1
        v[~np.isin(lay, ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L2/3/4', 'WM'])] = np.nan
        dep[m] = v
    return dep
