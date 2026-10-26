"""Small synthetic dataset in the layout of the working directory, for the unit tests.

Six sections T401-T406, each a 64 x 24 grid (step 100): x runs along the cortex, y through the depth.
  layers  : y bands L1 (0-1), L2 (2-5), L3 (6-9), L4 (10-12), L5 (13-16), L6 (17-20), WM (21-23)
  regions : 16 regions R01-R16 of 4 grid steps along x; columns = the same 4-step bins (L2-L6 spots), 16 per section
  areas   : R01-R04 -> 'A', R05-R08 -> 'B', R09-R12 -> 'C', R13-R16 -> 'D'
  truth   : three orthonormal region components mu_1..3 (mu_1 also drives the column SA index), gene and latent loadings per
            layer; amplitudes 1.0, 0.7, 0.5
  counts  : Poisson(library x softmax(b + layer profile + section offset + sum_m a_m mu_m(r) w_m)), 60 genes
  latents : 'truth' (16 dims): layer profile + section offset + sum_m a_m mu_m(r) g_ml + N(0, 1) noise
            'random' : layer profile + N(0, 1) noise (no areal structure)
            'curve_r1.5_centred' : = truth, used as the FGW anchor space
make(root) writes inputs/, embeddings/ and a config yaml under root and returns (config path, truth dict)."""
import os, json
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

DS = 'pfcserSyn'; NX, NY = 64, 24; SECTIONS = [f'T40{i}' for i in range(1, 7)]; NREG = 16; G = 60; F = 16
BANDS = [('L1', 0, 2), ('L2', 2, 6), ('L3', 6, 10), ('L4', 10, 13), ('L5', 13, 17), ('L6', 17, 21), ('WM', 21, 24)]
LAYERS = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']; AMP = np.array([1.0, 0.7, 0.5])


def layer_of(y):
    for l, a, b in BANDS:
        if a <= y < b: return l


def make(root, seed=0):
    rng = np.random.default_rng(seed); os.makedirs(f'{root}/inputs', exist_ok=True); os.makedirs(f'{root}/embeddings/{DS}', exist_ok=True)
    regs = [f'R{i + 1:02d}' for i in range(NREG)]
    M = rng.standard_normal((NREG, 3)); M -= M.mean(0); Q, _ = np.linalg.qr(M); MU = Q * np.sqrt(NREG)          # orthonormal, unit variance
    W = rng.standard_normal((3, G)); b = rng.normal(-np.log(G), 0.5, G); Lp = {l: rng.normal(0, 0.5, G) for l in LAYERS + ['WM']}
    Gl = {l: rng.standard_normal((3, F)) for l in LAYERS + ['WM']}; Lz = {l: rng.normal(0, 1.0, F) for l in LAYERS + ['WM']}
    rows, X, Zt, Zr = [], [], [], []
    for si, s in enumerate(SECTIONS):
        off_g = rng.normal(0, 0.1, G); off_z = rng.normal(0, 0.3, F)
        for x in range(NX):
            r = x // 4; ri = r
            for y in range(NY):
                l = layer_of(y); mu = MU[ri] * AMP; lib = rng.integers(200, 400)
                lg = b + Lp[l] + off_g + mu @ W; p = np.exp(lg - lg.max()); p /= p.sum(); X.append(rng.poisson(lib * p))
                Zt.append(Lz[l] + off_z + mu @ Gl[l] + rng.standard_normal(F)); Zr.append(Lz[l] + off_z + rng.standard_normal(F))
                cortex = l in ('L2', 'L3', 'L4', 'L5', 'L6')
                d = 0.0 if l == 'L1' else (1.0 if l == 'L6' else (1.1 if l == 'WM' else (y - 1.5) / 18.0))
                rows.append(dict(spot_id=f'{s}_{x}_{y}-{si}', slice=s, slice_full=f'{s}_adult_Sub-CTX', group='PFC', animal=s, region=regs[ri], layer=l,
                                 library_size=float(lib), col=(x // 4 + 1) if cortex else -1, arc=float(x * 100), depth=d, x=x * 100.0, y=y * 100.0))
    O = pd.DataFrame(rows).set_index('spot_id', drop=False); Xc = sp.csr_matrix(np.array(X, np.float32))
    O['library_size'] = np.asarray(Xc.sum(1)).ravel().clip(1)
    A = ad.AnnData(Xc, obs=O.drop(columns=['x', 'y']), var=pd.DataFrame(index=[f'G{g:02d}' for g in range(G)])); A.obsm['spatial'] = O[['x', 'y']].values.astype(np.float32)
    A.write_h5ad(f'{root}/inputs/{DS}.h5ad')
    C = O[O.col >= 0].groupby(['slice', 'col']).agg(area_level3=('region', lambda v: v.value_counts().index[0]), mean_x=('x', 'mean'), mean_y=('y', 'mean')).reset_index()
    C['area2'] = [('A', 'B', 'C', 'D')[(int(r[1:]) - 1) // 4] for r in C.area_level3]; C['SA'] = 'unassigned'
    C['Ridge_SA_index'] = [MU[regs.index(r), 0] + rng.normal(0, 0.2) for r in C.area_level3]
    C.to_csv(f'{root}/inputs/{DS}_columns.csv', index=False)
    for name, Z in (('truth', Zt), ('random', Zr), ('curve_r1.5_centred', Zt)):
        np.save(f'{root}/embeddings/{DS}/{name}__seed0.npy', np.array(Z, np.float32)); json.dump({}, open(f'{root}/embeddings/{DS}/{name}__seed0.json', 'w'))
    cfgp = f'{root}/config.yaml'
    open(cfgp, 'w').write(f'paths:\n  work: {root}\n  results: {root}/results\nseries:\n  name: {DS}\n  sections: {SECTIONS}\n')
    return cfgp, dict(mu=pd.DataFrame(MU, index=regs, columns=['mu1', 'mu2', 'mu3']))
