"""Mean-of-logs vs pooled (sum counts, divide by sum of library sizes, then log) on the simulations SimR / SimK.

Part A - pseudobulk axes in gene space (as the sim test 'Expression'):
  meanlog : block value = mean over spots of log1p(1e4 x / L)          (current)
  pooled  : block value = log1p(1e4 * sum x / sum L)
  both divided by the same per-gene SD (SD over spots of log1p CP10k), then exactly the steps of multi_axes.pb_axes.
Part B - smoothed PCA r 1.5 (PCA-32, centred per section) -> unchanged ottr.areal.multi_axes.run:
  disk_meanlog : disk mean of log1p CP10k                               (current smoothed PCA)
  disk_pooled  : log1p(1e4 * disk sum x / disk sum L)
Usage: OTTR_CONFIG=<scratch config> python pooled_test.py <out dir>"""
import os, sys, contextlib, io
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from sklearn.decomposition import PCA
from ottr.config import cfg, ensure
from ottr.utils.smoothing import grid_positions, disk_mean
from ottr.areal import multi_axes

OUT = sys.argv[1]; c = cfg()
LAYERS = multi_axes.LAYERS; BAD = multi_axes.BAD; K = 5
TR = pd.read_csv(c.tables('sim/truth_regions.csv'), index_col=0); TRUE = list(TR.columns)


def adj_r2(y, X):
    X1 = np.c_[np.ones(len(y)), X]; b, *_ = np.linalg.lstsq(X1, y, rcond=None); r2 = 1 - ((y - X1 @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    n, p = X.shape; return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def pb_fit(PB, W, kt, kr, kl, lib, sec, SEC):
    """multi_axes.pb_axes on precomputed block values PB (blocks x features): layer centring, theta c_t, region x (layer, f) SVD."""
    PB = PB.copy(); F = PB.shape[1]
    for l in LAYERS:
        q = kl == l
        if q.any(): PB[q] -= np.average(PB[q], axis=0, weights=W[q])
    cc_ = {s: np.log(np.median(lib[sec == s])) for s in SEC}; cmn = np.mean(list(cc_.values())); ct = np.array([cc_[t] - cmn for t in kt])
    th = (W * ct) @ PB / (W * ct * ct).sum(); PB -= ct[:, None] * th[None, :]
    regs = sorted(set(kr)); M = np.zeros((len(regs), len(LAYERS) * F)); hv = np.zeros((len(regs), len(LAYERS)), bool)
    for i, r in enumerate(regs):
        for j, l in enumerate(LAYERS):
            q = (kr == r) & (kl == l)
            if q.any(): M[i, j * F:(j + 1) * F] = np.average(PB[q], axis=0, weights=W[q]); hv[i, j] = True
    keep = hv.sum(1) >= 3; regs = [r for r, k in zip(regs, keep) if k]; M = M[keep]; M -= M.mean(0)
    U, s_, Vt = np.linalg.svd(M, full_matrices=False); Kp = min(K, len(regs) - 1)
    return pd.DataFrame(U[:, :Kp] * np.sqrt(len(regs)), index=regs), s_ ** 2 / (s_ ** 2).sum()


def splithalf(mu, mo, me):
    out = []
    for j in mu.columns:
        f = mu[j]; ci = mo.index.intersection(f.index); ce = me.index.intersection(f.index); cc_ = mo.index.intersection(me.index)
        bo = max(mo.columns, key=lambda c_: abs(np.corrcoef(mo.loc[ci, c_], f[ci])[0, 1])); be = max(me.columns, key=lambda c_: abs(np.corrcoef(me.loc[ce, c_], f[ce])[0, 1]))
        out.append(abs(np.corrcoef(mo.loc[cc_, bo], me.loc[cc_, be])[0, 1]))
    return out


def evaluate(mu, ve, sh, ds, variant, depth, sa_reg, rows, trows, rng):
    ix = mu.index.intersection(TR.index); T_ = TR.loc[ix]; Mu = mu.loc[ix]
    for k in mu.columns:
        r = {tc: abs(np.corrcoef(Mu[k], T_[tc])[0, 1]) for tc in TRUE}
        sar = sa_reg.reindex(mu.index); q = sar.notna()
        rows.append(dict(data=ds, variant=variant, comp=k + 1, var_share=ve[k], splithalf=sh[k], SA_r_region=abs(np.corrcoef(mu[k][q], sar[q])[0, 1]),
                         depth_r=abs(np.corrcoef(mu[k], depth.reindex(mu.index))[0, 1]), **{f'r_{tc}': r[tc] for tc in TRUE}))
    null = np.percentile([adj_r2(rng.standard_normal(len(ix)), Mu.values) for _ in range(200)], 95)
    for tc in TRUE: trows.append(dict(data=ds, variant=variant, truth=tc, adjR2=adj_r2(T_[tc].values, Mu.values), null95=null))


rows, trows = [], []
for ds in ('pfcserLSimR', 'pfcserLSimK'):
    rng = np.random.default_rng(0)
    A = ad.read_h5ad(c.inputs(ds)); o = A.obs; Y = A.X.tocsr().astype(np.float32)
    Ls = np.maximum(np.asarray(Y.sum(1)).ravel(), 1)                 # per-spot total of the simulated counts (as expr_logcp10k)
    sec = o.slice.astype(str).values; lay = o.layer.astype(str).values; reg = o.region.astype(str).values; lib = o.library_size.values.astype(float)
    xy = np.asarray(A.obsm['spatial'], float); SEC = sorted(set(sec), key=lambda t: int(t[1:]))
    Lg = (sp.diags(1e4 / Ls) @ Y).tocsr(); Lg.data = np.log1p(Lg.data); Lg = Lg.toarray()
    sd = Lg.std(0); keep = sd > 0; Lg = Lg[:, keep]; sd = sd[keep]; Yd = Y[:, np.flatnonzero(keep)].toarray()
    C = pd.read_csv(c.columns(ds)); sa_reg = C.groupby('area_level3').Ridge_SA_index.mean()
    # region depth after the model's own corrections (section median, layer centring) - as in depth_check.py
    okl = np.isin(lay, LAYERS); okr = ~np.isin(reg, BAD)
    res = np.log(Ls) - pd.Series(np.log(Ls)).groupby(sec).transform(lambda v: np.log(np.median(np.exp(v)))).values
    dfd = pd.DataFrame(dict(r=reg, l=lay, res=res))[okl]; dfd['res'] -= dfd.groupby('l').res.transform('mean'); depth = dfd.groupby('r').res.mean()

    # ---------- Part A: pseudobulks in gene space
    rcount = pd.Series(reg[okl & okr]).value_counts(); GOODREG = list(rcount[rcount >= 30].index)

    def blocks(keep_sec):
        mm = okl & okr & np.isin(sec, keep_sec) & np.isin(reg, GOODREG)
        gg = pd.DataFrame({'t': sec[mm], 'r': reg[mm], 'l': lay[mm]}).groupby(['t', 'r', 'l']).indices; ii = np.flatnonzero(mm)
        kk = [k for k, v in gg.items() if len(v) >= 5]; ix = [ii[gg[k]] for k in kk]
        W = np.array([len(v) for v in ix], float); kt, kr, kl = (np.array([k[j] for k in kk]) for j in range(3))
        meanlog = np.array([Lg[v].mean(0) for v in ix]) / sd
        pooled = np.array([np.log1p(1e4 * Yd[v].sum(0) / Ls[v].sum()) for v in ix]) / sd
        return dict(meanlog=meanlog, pooled=pooled), W, kt, kr, kl

    odd = [s for s in SEC if int(s[1:]) % 2]; even = [s for s in SEC if not int(s[1:]) % 2]
    BA, BO, BE = blocks(SEC), blocks(odd), blocks(even)
    for v in ('meanlog', 'pooled'):
        mu, ve = pb_fit(BA[0][v], *BA[1:], lib, sec, SEC); mu = (mu - mu.mean()) / mu.std()
        mo, _ = pb_fit(BO[0][v], *BO[1:], lib, sec, SEC); me, _ = pb_fit(BE[0][v], *BE[1:], lib, sec, SEC)
        evaluate(mu, ve, splithalf(mu, mo, me), ds, f'pseudobulk_{v}', depth, sa_reg, rows, trows, rng)
        print(ds, 'pseudobulk', v, 'done', flush=True)

    # ---------- Part B: smoothed PCA r 1.5 -> multi_axes (unchanged code)
    G = grid_positions(sec, xy)
    Xm, _ = disk_mean(lambda idx: Lg[idx], Lg.shape[1], G, 1.5, chunk=200)
    Xc, _ = disk_mean(lambda idx: Yd[idx], Yd.shape[1], G, 1.5, chunk=200)
    Ld, _ = disk_mean(lambda idx: Ls[idx][:, None], 1, G, 1.5, chunk=1)
    Xp = np.log1p(1e4 * Xc / Ld)                                     # disk mean counts / disk mean L = disk sum / disk sum
    del Xc
    E = ensure(c.emb_dir(ds))
    for v, X in (('disk_meanlog', Xm), ('disk_pooled', Xp)):
        P = PCA(32, random_state=0).fit_transform(X)
        for s in np.unique(sec): m = sec == s; P[m] -= P[m].mean(0)
        np.save(c.emb(ds, v), P.astype(np.float32))
    del Xm, Xp
    with contextlib.redirect_stdout(io.StringIO()):
        multi_axes.run([('disk_meanlog', 'disk_meanlog'), ('disk_pooled', 'disk_pooled')], ds=ds, tag='_pool')
    for v in ('disk_meanlog', 'disk_pooled'):
        Z = np.load(os.path.join(multi_axes.axes_dir(), f'{ds}_pool_{v}.npz'), allow_pickle=True)
        T = pd.read_csv(c.tables(f'axes_{ds}_pool_comp.csv')); T = T[(T.model == v) & (T.analysis == 'pseudobulk')]
        mu = pd.DataFrame(Z['pb_mu'], index=Z['pb_regions'])
        evaluate(mu, T.var_share.values, T.splithalf.values, ds, v, depth, sa_reg, rows, trows, rng)
        rows[-5:] = [dict(r, reading=rd) for r, rd in zip(rows[-5:], T.reading.values)]
    print(ds, 'smoothed PCA done', flush=True)

A_ = pd.DataFrame(rows); B_ = pd.DataFrame(trows)
A_.to_csv(f'{OUT}/axes.csv', index=False); B_.to_csv(f'{OUT}/truth.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(A_.round(3).to_string()); print(B_.round(3).to_string())
