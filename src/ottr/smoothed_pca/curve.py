"""Smoothed PCA - the training-free reference and the anchor space of FGW (step 02).
python -m ottr.smoothed_pca.curve [--ds pfcserL] [--radii 0,1.5,2.5,3.5,5,7]

Idea: averaging expression over a small disk removes spot-level sampling noise while keeping layers (a disk of radius 1.5
grid steps = 9 spots, ~75 um, is thinner than a cortical layer). PCA (32 components, scikit-learn, random_state 0) of the
smoothed log1p CP10k gives a latent with no training; three section corrections are stored:
  none      PCA scores as they are
  centred   minus the per-section mean (removes section offsets)       <- used as the FGW anchor space (r = 1.5)
  harmony   Harmony on the PCA scores, batch = section
Output: embeddings/<ds>/curve_r<r>_<variant>__seed0.npy (+ .json). The radius is chosen by ottr.smoothed_pca.select.

smoothed_pca_sd(): variant used for the young brains (the anchor space of their FGW spec): every gene divided by its SD
before smoothing, PCA by SVD on every third spot, centred per section -> smoothed_pca32{,_centred}__seed0.npy."""
import os, json, argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from ..config import cfg, ensure
from ..utils.smoothing import grid_positions, disk_mean

RADII = [0, 1.5, 2.5, 3.5, 5, 7, 10, 15, 20, 25]


def lognorm(A):
    """log1p(1e4 x / library_size) as CSR float32 (library = all genes of the section, obs.library_size)."""
    L = sp.diags(1e4 / A.obs.library_size.values.astype(np.float32)) @ A.X; L.data = np.log1p(L.data)
    return L.tocsr().astype(np.float32)


def curve(ds, radii=RADII, variants=('none', 'centred', 'harmony')):
    c = cfg(); E = ensure(c.emb_dir(ds)); A = ad.read_h5ad(c.inputs(ds)); L = lognorm(A)
    sl = A.obs.slice.astype(str).values; xy = np.asarray(A.obsm['spatial'], float); G = grid_positions(sl, xy); info = {}
    for r in radii:
        if r == 0: X = L.toarray(); nn = 1.0
        else: X, nn = disk_mean(lambda idx: L[idx].toarray(), L.shape[1], G, r, chunk=200)
        P = PCA(32, random_state=0).fit_transform(X)
        out = {'none': P}
        if 'centred' in variants:
            C = P.copy()
            for s in np.unique(sl): m = sl == s; C[m] -= C[m].mean(0)
            out['centred'] = C
        if 'harmony' in variants:
            import harmonypy
            H = np.asarray(harmonypy.run_harmony(P, pd.DataFrame({'slice': sl}), 'slice', random_state=0, verbose=False).Z_corr)
            out['harmony'] = H.T if H.shape[0] != A.n_obs else H
        for v in variants:
            name = f'curve_r{r:g}_{v}__seed0'; np.save(f'{E}/{name}.npy', out[v].astype(np.float32))
            json.dump(dict(kind='smoothing_curve', radius_steps=r, radius_um=r * 50, mean_spots_in_disk=nn, variant=v, dim=32), open(f'{E}/{name}.json', 'w'))
        info[r] = nn; print(ds, 'radius', r, 'mean spots in disk', round(nn, 1), flush=True)
    return info


def smoothed_pca_sd(ds, r=1.5):
    c = cfg(); E = ensure(c.emb_dir(ds)); A = ad.read_h5ad(c.inputs(ds)); sec = A.obs.slice.astype(str).values
    X = lognorm(A).toarray(); X /= X.std(0) + 1e-9
    S, _ = disk_mean(lambda idx: X[idx], X.shape[1], grid_positions(sec, np.asarray(A.obsm['spatial'], float)), r, chunk=X.shape[1], min_count=1)
    Sc = S - S.mean(0); _, _, Vt = np.linalg.svd(Sc[::3], full_matrices=False); P = Sc @ Vt[:32].T
    for name, Z in [('smoothed_pca32', P), ('smoothed_pca32_centred', P - np.vstack([P[sec == s].mean(0) for s in sec]))]:
        np.save(f'{E}/{name}__seed0.npy', Z.astype(np.float32)); json.dump(dict(kind='baseline', dim=32), open(f'{E}/{name}__seed0.json', 'w'))
    print(ds, 'smoothed PCA (gene SD) saved', P.shape)


if __name__ == '__main__':
    P_ = argparse.ArgumentParser(); P_.add_argument('--ds', default=cfg().series['name']); P_.add_argument('--radii', default=None)
    P_.add_argument('--variants', default='none,centred,harmony'); P_.add_argument('--sd', action='store_true', help='young-brain variant')
    a = P_.parse_args()
    if a.sd: smoothed_pca_sd(a.ds)
    else: curve(a.ds, [float(x) for x in a.radii.split(',')] if a.radii else RADII, tuple(a.variants.split(',')))
