"""E7, E8, E10 of the validation plan: connectivity and morphometry of the 30 PFC regions against our axes.   python -m ottr.validation.connect
E7  MBMv4 resting-state FC (Tian et al. 2022; connectome_data.mat, Paxinos 116 regions x 2 hemispheres, intra-hemispheric blocks averaged)
E8  MBMv4 dMRI structural connectivity (same file) and retrograde tracing (Majka et al. 2020, FLN target x source).
    primary (15): mu axis k x {FC, SC, FLN} Mantel: |difference of mu| between PFC region pairs vs dissimilarity of their profiles
        (1 - Spearman of the profile over all other regions, the two regions themselves left out; FLN: profile of a region as SOURCE
        over the 55 injected targets, log10(FLN + 1e-7))
    exploratory: diffusion-map gradients G1-G3 of the whole-cortex FC / SC (top 10 % per row, cosine affinity, alpha 0.5) at the 30
        regions; Mantel over the 14 PFC regions that were injected (their input profiles over the 116 sources)
E10 BMA2.0 surfaces (left + right averaged): surface area (graymid), thickness (white -> pial distance), mean curvature (graymid,
    cotangent Laplacian), sulcal depth (pial distance to the convex hull); exploratory, 5 axes x 4 + gene scores
-> results/tables/validation/rows_connectivity.csv, rows_morphometry.csv, results/tables/validation/morphometry_regions.csv"""
import os, re
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import rankdata
from scipy.spatial import ConvexHull
from . import common as C, gifti

MBM = '/workspace/external_repos/mbm_scidb/connectome_data'; MAJ = '/workspace/external_repos/mbca_majka2020'; BMA = '/workspace/external_repos/bma2.0'


def paxinos():
    m = sio.loadmat(f'{MBM}/connectome_data.mat'); L = pd.read_csv(f'{MBM}/Paxinos_atlas_labels.csv'); ab = L.abbre.str.strip().tolist()
    h = lambda M: (M[:116, :116] + M[116:, 116:]) / 2
    return pd.DataFrame(h(m['Paxinos_FC']), index=ab, columns=ab), pd.DataFrame(h(m['Paxinos_SC']), index=ab, columns=ab)


def name_map(names, regions):
    low = {n.lower().replace('/', '-'): n for n in names}; return {r: low.get(r.lower().replace('/', '-')) for r in regions}


def profile_dissim(P, rows):
    """P: units x features (NaN allowed); 1 - Spearman of two rows over the features both have, the two units themselves dropped."""
    n = len(rows); D = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            keep = [c for c in P.columns if c not in (rows[i], rows[j])]; a, b = P.loc[rows[i], keep].values, P.loc[rows[j], keep].values; f = np.isfinite(a) & np.isfinite(b)
            D[i, j] = D[j, i] = 1 - np.corrcoef(rankdata(a[f]), rankdata(b[f]))[0, 1]
    return D


def diffusion_map(A, k=3, alpha=0.5, keep=0.1):
    A = np.asarray(A, float).copy(); np.fill_diagonal(A, 0)
    thr = np.quantile(A, 1 - keep, axis=1, keepdims=True); A[A < thr] = 0; A[A < 0] = 0
    nrm = np.linalg.norm(A, axis=1, keepdims=True); nrm[nrm == 0] = 1; K = (A / nrm) @ (A / nrm).T
    d = K.sum(1); K = K / np.outer(d ** alpha, d ** alpha); P = K / K.sum(1, keepdims=True)
    w, v = np.linalg.eig(P); o = np.argsort(-w.real); w, v = w.real[o], v.real[:, o]
    return v[:, 1:k + 1] * (w[1:k + 1] / (1 - w[1:k + 1]))


def connectivity(rng):
    O = C.ours(); regs = list(O.index); FC, SC = paxinos(); mp = name_map(FC.index, regs); assert all(mp.values()), mp; pr = [mp[r] for r in regs]
    fl = pd.read_csv(f'{MAJ}/marmoset_brain_connectivity_1_0_fln_matrix.txt', sep=' ', header=None, names=['target', 'source', 'fln'])
    F = fl.pivot_table(index='target', columns='source', values='fln'); F = np.log10(F + 1e-7); ms = name_map(F.columns, regs); mt = name_map(F.index, regs)
    rows = []; prof = {'FC (MBMv4 resting state)': FC, 'SC (MBMv4 dMRI tractography)': SC, 'FLN outputs (Majka 2020 tracing)': F.T}
    D = {}
    for nm, P in prof.items():
        units = pr if not nm.startswith('FLN') else [ms[r] for r in regs]
        D[nm] = profile_dissim(P, units)
    for k in range(1, 6):
        for nm in prof: rows.append(C.mantel(f'mu axis{k}', D[nm], regs, rng=rng, domain='connectivity', dataset=nm, species='marmoset', feature='Mantel: profile dissimilarity', family='primary',
                                           note='rho = Spearman over region pairs of |difference of mu| vs 1 - profile similarity'))
    for nm, M in (('FC (MBMv4 resting state)', FC), ('SC (MBMv4 dMRI tractography)', SC)):
        G = pd.DataFrame(diffusion_map(M.values), index=M.index, columns=['G1', 'G2', 'G3']).loc[pr]
        for o in O:
            for g in G: rows.append(C.test(o, G[g].values, rng=rng, domain='connectivity', dataset=nm, species='marmoset', feature=f'diffusion gradient {g} (whole cortex)', family='exploratory', note='gradient sign arbitrary'))
    inj = [r for r in regs if mt[r]]; Din = profile_dissim(F, [mt[r] for r in inj])
    for k in range(1, 6): rows.append(C.mantel(f'mu axis{k}', Din, inj, rng=rng, domain='connectivity', dataset='FLN inputs of injected PFC regions (Majka 2020)', species='marmoset', feature='Mantel: input profile dissimilarity',
                                             family='exploratory', note=f'descriptive: only {len(inj)} injected PFC regions'))
    return C.save(rows, 'connectivity')


def surface(h):
    P = dict(gifti.read(f'{BMA}/BMA2.0.{h}.pial.surf.gii')[0]); W = dict(gifti.read(f'{BMA}/BMA2.0.{h}.white.surf.gii')[0]); M = dict(gifti.read(f'{BMA}/BMA2.0.{h}.graymid.surf.gii')[0])
    A, lt = gifti.read(f'{BMA}/BMA2.0.{h}.regions.label.gii'); lab = A[0][1].astype(int); ab = {k: (re.search(r'\(([^)]+)\)\s*$', v).group(1) if re.search(r'\(([^)]+)\)\s*$', v) else v) for k, v in lt.items()}
    V, T = M['NIFTI_INTENT_POINTSET'].astype(float), M['NIFTI_INTENT_TRIANGLE'].astype(int); n = len(V)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]; tri_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1); varea = np.bincount(T.ravel(), np.repeat(tri_area / 3, 3), n)
    # mean curvature: cotangent Laplacian, H = 0.5 |L x| / A, signed by the outward normal
    L = np.zeros((n, 3)); nrm = np.zeros((n, 3)); fn = np.cross(b - a, c - a)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u, v = V[T[:, j]] - V[T[:, i]], V[T[:, k]] - V[T[:, i]]; cot = (u * v).sum(1) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-12)
        for p, q in ((j, k), (k, j)): np.add.at(L, T[:, p], 0.5 * cot[:, None] * (V[T[:, q]] - V[T[:, p]]))
        np.add.at(nrm, T[:, i], fn)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12); curv = 0.5 * (L * nrm).sum(1) / np.maximum(varea, 1e-12)
    Pp = P['NIFTI_INTENT_POINTSET'].astype(float); hull = ConvexHull(Pp); depth = np.min(-(Pp @ hull.equations[:, :3].T + hull.equations[:, 3]), axis=1)
    thick = np.linalg.norm(Pp - W['NIFTI_INTENT_POINTSET'].astype(float), axis=1)
    D = pd.DataFrame(dict(region=[ab.get(l, '') for l in lab], area=varea, thickness=thick, curvature=curv, sulcal_depth=depth))
    return D.groupby('region').apply(lambda g: pd.Series(dict(surface_area=g.area.sum(), thickness=np.average(g.thickness, weights=g.area), mean_curvature=np.average(g.curvature, weights=g.area),
                                                             sulcal_depth=np.average(g.sulcal_depth, weights=g.area), n_vertices=len(g))))


MBMS = '/workspace/external_repos/mbm_scidb/surfaces_fs'


def mbm_surface(h):
    """MBM in vivo MRI template surfaces (FreeSurfer-style, Liu et al. / MBMv4): the authors' thickness, pial / white curvature and
    myelin (T1w/T2w) maps, plus surface area (graymid) and sulcal depth (pial distance to the convex hull) computed here; Paxinos labels."""
    sh = lambda f: gifti.read(f'{MBMS}/{h}.{f}')[0][0][1].astype(float)
    A, lt = gifti.read(f'{MBMS}/{h}.MBM_cortex_vPaxinos.label.gii'); lab = A[0][1].astype(int)
    M = dict(gifti.read(f'{MBMS}/{h}.graymid.surf.gii')[0]); V, T = M['NIFTI_INTENT_POINTSET'].astype(float), M['NIFTI_INTENT_TRIANGLE'].astype(int)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]; varea = np.bincount(T.ravel(), np.repeat(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) / 3, 3), len(V))
    Pp = dict(gifti.read(f'{MBMS}/{h}.pial.surf.gii')[0])['NIFTI_INTENT_POINTSET'].astype(float); hull = ConvexHull(Pp)
    depth = np.min(-(Pp @ hull.equations[:, :3].T + hull.equations[:, 3]), axis=1)
    D = pd.DataFrame(dict(region=[lt.get(l, '') for l in lab], area=varea, thickness=sh('thickness.shape.gii'), pial_curv=sh('pial.curv.shape.gii'), white_curv=sh('white.curv.shape.gii'),
                          myelin=sh('myelinmap.shape.gii'), sulcal_depth=depth))
    D = D[~D.region.isin(['', 'background'])]
    return D.groupby('region')[[c for c in D if c != 'region']].apply(lambda g: pd.Series(dict(surface_area=g.area.sum(), **{k: np.average(g[k], weights=g.area) for k in ('thickness', 'pial_curv', 'white_curv', 'myelin', 'sulcal_depth')},
                                                             n_vertices=len(g))))


def mbm_morphometry(rng, hemis=('lh', 'rh')):
    O = C.ours(); Rs = [mbm_surface(h) for h in hemis]; R = sum(Rs) / len(Rs); mp = name_map(R.index, O.index)          # A14C and OPAl have no vertices on the MBM surface -> NaN
    R = R.reindex([mp[r] for r in O.index]).set_axis(O.index); R.to_csv(os.path.join(C.out_dir(), 'morphometry_mbm_regions.csv'))
    rows = [C.test(o, R[f].values, rng=rng, domain='morphometry', dataset=f'MBM MRI template surfaces ({"+".join(hemis)})', species='marmoset', feature=f,
                   family='exploratory' if o.startswith('mu') else 'gene scores') for o in O for f in ['thickness', 'pial_curv', 'white_curv', 'sulcal_depth', 'surface_area', 'myelin']]
    return C.save(rows, 'morphometry_mbm'), R


def morphometry(rng):
    O = C.ours(); R = (surface('lh') + surface('rh')) / 2; mp = name_map(R.index, O.index); R = R.loc[[mp[r] for r in O.index]].set_axis(O.index); R.to_csv(os.path.join(C.out_dir(), 'morphometry_regions.csv'))
    rows = [C.test(o, R[f].values, rng=rng, domain='morphometry', dataset='BMA2.0 surfaces', species='marmoset', feature=f, family='exploratory' if o.startswith('mu') else 'gene scores') for o in O for f in ['surface_area', 'thickness', 'mean_curvature', 'sulcal_depth']]
    return C.save(rows, 'morphometry'), R


if __name__ == '__main__':
    rng = np.random.default_rng(23); pd.set_option('display.width', 250)
    M, R = morphometry(rng); print(R.round(4).to_string()); print(M[M.ours.str.startswith('mu')][['ours', 'feature', 'spearman', 'p_perm', 'q_perm', 'p_msr', 'q_msr', 'p_vario', 'q_vario', 'verdict']].round(4).to_string(index=False))
    D = connectivity(rng); print(D[['dataset', 'ours', 'feature', 'n', 'spearman', 'p_perm', 'q_perm', 'p_msr', 'q_msr', 'p_vario', 'q_vario', 'verdict']].round(4).to_string(index=False))
