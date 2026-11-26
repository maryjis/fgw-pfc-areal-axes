"""Common tests of the validation plan: every comparison of OUR region map (mu of an axis, or a gene score, over the 30 adult PFC regions)
with a feature of another dataset gives one row of the same format (ROW_COLUMNS):

p_perm   plain permutation of the units (exact for n <= 8, else NPERM random permutations)
p_msr    spatial null: Moran spectral randomisation surrogates of OUR map (30 regions, 3D centroids), carried to the units of the dataset
         exactly as our map (mean over the mapped marmoset regions)
p_vario  spatial null: variogram-matched surrogates (BrainSMASH-like), carried the same way
Both spatial nulls keep the smoothness of our map; on independent smooth random maps of these regions they give p < 0.05 in 3-8 % of
cases, plain permutations in up to 23 % (results/tables/validation/null_power_sim.csv).
mantel() compares |difference of our map| between region pairs with a dissimilarity matrix (connectivity profiles), same three nulls.
finalize() adds BH q per (domain, family) for each p and the verdict:
  confirmed   q_msr < 0.05 and q_vario < 0.05, sign stable in leave-one-out, and the replication (if any) has the same sign
  nominal     p_msr < 0.05 and p_vario < 0.05 (no correction)
  perm only   q_perm < 0.05 but not nominal
  no          otherwise"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from statsmodels.stats.multitest import multipletests
from ..config import cfg, ensure
from ..genescore import significance as SG
from ..atlas import modalities as MD

NS, NPERM, NBOOT = 5000, 10000, 2000
ROW_COLUMNS = ['domain', 'dataset', 'species', 'axis', 'ours', 'feature', 'n', 'family', 'spearman', 'pearson', 'ci_low', 'ci_high', 'p_perm', 'q_perm',
               'p_msr', 'q_msr', 'p_vario', 'q_vario', 'p_extra', 'p_extra_kind', 'replication', 'loo_min', 'loo_max', 'note']


def out_dir():
    return ensure(cfg().tables('validation'))


def ours():
    """30 PFC regions x (mu axis1..5, genes PrAl, genes axis1..5)."""
    return MD.ours()


_SUR = {}


def surrogates(col):
    """{'msr', 'variogram'}: 30 x NS surrogates of our map `col` (values rank-matched to the original), cached in work/validation."""
    if col in _SUR: return _SUR[col]
    c = cfg(); d = ensure(os.path.join(c.work, 'validation')); f = os.path.join(d, 'surrogates_' + col.replace(' ', '_') + '.npz')
    if os.path.exists(f):
        z = np.load(f); _SUR[col] = {k: z[k] for k in z.files}; return _SUR[col]
    O = ours(); x = O[col].values.astype(float); C = SG.marmoset_centroids(list(O.index)); rng = np.random.default_rng(abs(hash(col)) % 2 ** 32)
    S = {m: SG.surrogates(x, C, NS, rng, m)[0] for m in ('msr', 'variogram')}; np.savez(f, **S); _SUR[col] = S; return S


def carry(values30, mapping):
    """values over the 30 regions (Series or 30 x k array with the index of ours()) -> units: mean over the mapped regions (list of lists)."""
    O = ours(); pos = {r: i for i, r in enumerate(O.index)}; V = np.asarray(values30, float)
    if V.ndim == 1: V = V[:, None]
    out = np.full((len(mapping), V.shape[1]), np.nan)
    for i, regs in enumerate(mapping):
        ix = [pos[r] for r in regs if r in pos]
        if ix: out[i] = np.nanmean(V[ix], 0)
    return out


def _perm_null(x, y, rng):
    n = len(x); rx, ry = SG.unit(x), SG.unit(y)
    if n <= 8: P = np.array(list(itertools.permutations(range(n))))
    else: P = np.array([rng.permutation(n) for _ in range(NPERM)])
    return rx[P] @ ry


def residual_ranks(v, Z):
    """ranks of v with the ranks of the covariates Z (n x k) regressed out (partial Spearman)."""
    R = np.column_stack([np.ones(len(v))] + [rankdata(Z[:, j]) for j in range(Z.shape[1])]); r = rankdata(v); return r - R @ np.linalg.lstsq(R, r, rcond=None)[0]


def test(col, y, mapping=None, covariates=None, rng=None, **meta):
    """Our map `col` (carried to the units by `mapping`; None = the 30 regions themselves) against feature y of the units."""
    rng = rng or np.random.default_rng(0); O = ours(); y = np.asarray(y, float)
    mapping = mapping or [[r] for r in O.index]; x = carry(O[col], mapping)[:, 0]; S = surrogates(col)
    ok = np.isfinite(x) & np.isfinite(y)
    if covariates is not None: ok &= np.isfinite(covariates).all(1)
    x, y = x[ok], y[ok]; n = len(x); Zc = covariates[ok] if covariates is not None else None
    if n < 5: return None
    f = (lambda v: residual_ranks(v, Zc)) if Zc is not None else (lambda v: v)
    xr, yr = f(x), f(y); r = float(np.corrcoef(rankdata(xr), rankdata(yr))[0, 1]) if Zc is None else float(np.corrcoef(xr, yr)[0, 1])
    null = _perm_null(xr, yr, rng) if Zc is None else np.array([np.corrcoef(xr, yr[rng.permutation(n)])[0, 1] for _ in range(NPERM)])
    row = dict(n=n, spearman=r, pearson=float(np.corrcoef(x, y)[0, 1]), p_perm=SG.p_two(null, r))
    for m, key in (('msr', 'p_msr'), ('variogram', 'p_vario')):
        Sm = carry(S[m], mapping)[ok]
        if Zc is None: nl = SG.rank_cols(Sm).T @ SG.unit(y)
        else: nl = np.array([np.corrcoef(f(Sm[:, j]), yr)[0, 1] for j in range(Sm.shape[1])])
        row[key] = SG.p_two(nl, r)
    if n >= 6:
        bi = rng.integers(0, n, (NBOOT, n)); bs = []
        for i in bi:
            if len(set(i)) < 4: continue
            a, b = f(x[i]) if Zc is None else xr[i], f(y[i]) if Zc is None else yr[i]; bs.append(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
        bs = np.array(bs); bs = bs[np.isfinite(bs)]; row.update(ci_low=float(np.quantile(bs, .025)), ci_high=float(np.quantile(bs, .975)))
        loo = [np.corrcoef(rankdata(np.delete(xr, i)), rankdata(np.delete(yr, i)))[0, 1] for i in range(n)]; row.update(loo_min=float(min(loo)), loo_max=float(max(loo)))
    row.update(ours=col, axis=col.split()[-1].replace('axis', '') if 'axis' in col else 'PrAl', **meta); return row


def mantel(col, D, regions, rng=None, **meta):
    """|difference of our map| over region pairs vs dissimilarity D (regions x regions, same order as `regions`); Spearman over pairs."""
    rng = rng or np.random.default_rng(0); O = ours(); pos = {r: i for i, r in enumerate(O.index)}; ix = np.array([pos[r] for r in regions]); x = O[col].values.astype(float)[ix]
    iu = np.triu_indices(len(ix), 1); d = np.asarray(D, float)[iu]; ok = np.isfinite(d); rd = SG.unit(d[ok])
    stat = lambda v: float(SG.unit(np.abs(v[:, None] - v[None, :])[iu][ok]) @ rd)
    r = stat(x); n = len(ix); null = np.array([stat(x[rng.permutation(n)]) for _ in range(NPERM // 2)]); S = surrogates(col)
    row = dict(n=n, spearman=r, pearson=np.nan, p_perm=SG.p_two(null, r), ours=col, axis=col.split()[-1].replace('axis', '') if 'axis' in col else 'PrAl', **meta)
    for m, key in (('msr', 'p_msr'), ('variogram', 'p_vario')): row[key] = SG.p_two(np.array([stat(S[m][ix, j]) for j in range(S[m].shape[1])]), r)
    return row


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); f = np.isfinite(p)
    if f.any(): q[f] = multipletests(p[f], method='fdr_bh')[1]
    return q


def finalize(D):
    D = D.copy()
    for c in ROW_COLUMNS:
        if c not in D: D[c] = np.nan
    for p in ('p_perm', 'p_msr', 'p_vario'):
        q = 'q' + p[1:]; D[q] = np.nan
        for _, i in D.groupby(['domain', 'family'], dropna=False).groups.items(): D.loc[i, q] = bh(D.loc[i, p])
    loo_ok = ~((np.sign(D.loo_min) != np.sign(D.spearman)) | (np.sign(D.loo_max) != np.sign(D.spearman))) | D.loo_min.isna()
    rep_ok = D.replication.isna() | (D.replication.astype(str) != 'opposite')
    conf = (D.q_msr < .05) & (D.q_vario < .05) & loo_ok & rep_ok; nom = (D.p_msr < .05) & (D.p_vario < .05)
    D['verdict'] = np.select([conf, nom, D.q_perm < .05], ['confirmed', 'nominal', 'perm only'], 'no'); return D[ROW_COLUMNS + ['verdict']]


def save(rows, name):
    D = finalize(pd.DataFrame([r for r in rows if r is not None])); D.to_csv(os.path.join(out_dir(), f'rows_{name}.csv'), index=False); return D


def test_best(col, Y, names, mapping=None, rng=None, **meta):
    """Our map against the best of several components (columns of Y): |rho| of the best one, every null takes the maximum over the components."""
    rng = rng or np.random.default_rng(0); O = ours(); Y = np.asarray(Y, float); mapping = mapping or [[r] for r in O.index]
    x = carry(O[col], mapping)[:, 0]; ok = np.isfinite(x) & np.isfinite(Y).all(1); x, Y = x[ok], Y[ok]; n = len(x)
    U = np.column_stack([SG.unit(Y[:, j]) for j in range(Y.shape[1])]); R = SG.unit(x) @ U; b = int(np.abs(R).argmax())
    P = np.array(list(itertools.permutations(range(n)))) if n <= 8 else np.array([rng.permutation(n) for _ in range(NPERM)])
    nullp = np.abs(SG.unit(x)[P] @ U).max(1); row = dict(n=n, spearman=float(R[b]), pearson=float(np.corrcoef(x, Y[:, b])[0, 1]), p_perm=(1 + (nullp >= abs(R[b]) - 1e-12).sum()) / (len(P) + 1))
    for m, key in (('msr', 'p_msr'), ('variogram', 'p_vario')):
        Sm = SG.rank_cols(carry(surrogates(col)[m], mapping)[ok]); nl = np.abs(Sm.T @ U).max(1); row[key] = (1 + (nl >= abs(R[b]) - 1e-12).sum()) / (len(nl) + 1)
    row.update(ours=col, axis=col.split()[-1].replace('axis', '') if 'axis' in col else 'PrAl', best=names[b], all_rho='; '.join(f'{nm} {r:+.2f}' for nm, r in zip(names, R)), **meta); return row
