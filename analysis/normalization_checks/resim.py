"""Re-draw the random simulation (SimR) with new seeds and see which axes come out.
  real depth  : seeds 1, 2, 3 - counts ~ Multinomial(real per-spot total, real mean gene proportions) (as ottr.areal.simulation.make)
  equal depth : seed 1 with every spot's total = the median total (no depth structure at all)
Per data set: (a) the project's model on log1p CP10k / gene SD (multi_axes.run unchanged = mean of logs in the pseudobulks);
              (b) pooled pseudobulks log1p(1e4 sum x / sum L) with the same fit and the same reading rules.
Usage: OTTR_CONFIG=<scratch config> python resim.py <out dir>"""
import os, sys, contextlib, io
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from ottr.config import cfg, ensure
from ottr.areal import multi_axes

OUT = sys.argv[1]; c = cfg(); LAYERS = multi_axes.LAYERS; BAD = multi_axes.BAD; K = 5
src = open(os.path.join(OUT, 'pooled_test.py')).read(); exec(src[src.index('def pb_fit'):src.index('def evaluate')])  # pb_fit, splithalf

base = ad.read_h5ad(c.inputs('pfcserLSimR')); o = base.obs.copy(); xy = np.asarray(base.obsm['spatial'])
R0 = ad.read_h5ad('/workspace/Gene_modules/optimal_transport__tr_axeses/work/inputs/pfcserL.h5ad', backed='r')   # real counts of the model genes
Xr = R0.X[:].tocsr(); genes = np.array(R0.var_names)
p0 = np.asarray(Xr.sum(0)).ravel() + 1; p0 /= p0.sum()                       # as simulation.make
tot = np.asarray(Xr.sum(1)).ravel().astype(np.int64); tot[tot < 1] = 1
sec = o.slice.astype(str).values; lay = o.layer.astype(str).values; reg = o.region.astype(str).values; lib = o.library_size.values.astype(float)
SEC = sorted(set(sec), key=lambda t: int(t[1:])); snum = np.array([int(s[1:]) for s in sec])
C = pd.read_csv(c.columns('pfcserLSimR')); sa_reg = C.groupby('area_level3').Ridge_SA_index.mean()
okl = np.isin(lay, LAYERS); okr = ~np.isin(reg, BAD); rc = pd.Series(reg[okl & okr]).value_counts(); GOOD = list(rc[rc >= 30].index)
rcn = pd.Series(snum[okl & okr]).groupby(reg[okl & okr]).mean()
res = np.log(tot) - pd.Series(np.log(tot)).groupby(sec).transform(lambda v: np.log(np.median(np.exp(v)))).values
dd = pd.DataFrame(dict(r=reg, l=lay, res=res))[okl]; dd['res'] -= dd.groupby('l').res.transform('mean'); depth = dd.groupby('r').res.mean()


def reading(mu, sh):
    """The pseudobulk reading rules of multi_axes (SA > arch > rostro-caudal > not reproducible > areal axis)."""
    out = []
    for j in mu.columns:
        f = mu[j]; sar = sa_reg.reindex(f.index); q = sar.notna(); sr = np.corrcoef(f[q], sar[q])[0, 1]
        arch = np.nan
        if j > 0: b = np.polyfit(mu[0], f, 2); arch = 1 - ((f - np.polyval(b, mu[0])) ** 2).sum() / ((f - f.mean()) ** 2).sum()
        rcr = abs(np.corrcoef(f, rcn.reindex(f.index))[0, 1]); srt = f.sort_values()
        if abs(sr) >= 0.6: rd = 'SA axis'
        elif j > 0 and arch >= 0.5: rd = 'arch of axis 1'
        elif rcr >= 0.6: rd = 'rostro-caudal'
        elif sh[j] < 0.5: rd = f'not reproducible ({srt.index[0]} ↔ {srt.index[-1]})'
        else: rd = f'areal axis {srt.index[0]} ↔ {srt.index[-1]}'
        out.append((rd, sr))
    return out


def loo_splithalf(mo, me, mu):
    """Split-half without the single most influential region (min over leaving out each of the 3 most extreme regions)."""
    out = []
    for j in mu.columns:
        f = mu[j]; ci = mo.index.intersection(f.index); ce = me.index.intersection(f.index); cc_ = mo.index.intersection(me.index)
        bo = max(mo.columns, key=lambda c_: abs(np.corrcoef(mo.loc[ci, c_], f[ci])[0, 1])); be = max(me.columns, key=lambda c_: abs(np.corrcoef(me.loc[ce, c_], f[ce])[0, 1]))
        top = f.abs().sort_values(ascending=False).index[:3]
        out.append(min(abs(np.corrcoef(mo.loc[cc_.drop(t, errors='ignore'), bo], me.loc[cc_.drop(t, errors='ignore'), be])[0, 1]) for t in top))
    return out


rows = []
for name, seed, equal in [('seed1', 1, False), ('seed2', 2, False), ('seed3', 3, False), ('seed1_equal_depth', 1, True)]:
    rng = np.random.default_rng(seed); n = np.full_like(tot, int(np.median(tot))) if equal else tot
    Y = sp.vstack([sp.csr_matrix(rng.multinomial(n[i:i + 20000], p0).astype(np.float32)) for i in range(0, len(n), 20000)]).tocsr()
    ds = f'pfcserLSimR_{name}'; d_in = ensure(os.path.join(c.work, 'inputs'))
    A = ad.AnnData(Y, obs=o.copy(), var=pd.DataFrame(index=genes)); A.obsm['spatial'] = xy; A.write_h5ad(c.inputs(ds))
    if not os.path.exists(c.columns(ds)): os.symlink(c.columns('pfcserLSimR'), c.columns(ds))
    Ls = np.maximum(np.asarray(Y.sum(1)).ravel(), 1)
    Lg = (sp.diags(1e4 / Ls) @ Y).tocsr(); Lg.data = np.log1p(Lg.data); Lg = Lg.toarray(); sd = Lg.std(0); keep = sd > 0
    ensure(c.emb_dir(ds)); np.save(c.emb(ds, 'expr_logcp10k'), Lg[:, keep] / sd[keep])
    # (a) the project's model, unchanged
    with contextlib.redirect_stdout(io.StringIO()):
        T = multi_axes.run([('meanlog', 'expr_logcp10k')], ds=ds, tag='_rs')
    T = T[T.analysis == 'pseudobulk']
    Z = np.load(os.path.join(multi_axes.axes_dir(), f'{ds}_rs_meanlog.npz'), allow_pickle=True); mu_a = pd.DataFrame(Z['pb_mu'], index=Z['pb_regions'])
    for k, (_, t) in enumerate(T.iterrows()):
        rows.append(dict(data=name, variant='meanlog (project)', comp=k + 1, var_share=t.var_share, splithalf=t.splithalf, SA_r=abs(t.SA_r),
                         depth_r=abs(np.corrcoef(mu_a[k], depth.reindex(mu_a.index))[0, 1]), reading=t.reading))
    # (b) pooled pseudobulks
    Yd = Y[:, np.flatnonzero(keep)].toarray(); sdk = sd[keep]

    def blocks(keep_sec):
        mm = okl & okr & np.isin(sec, keep_sec) & np.isin(reg, GOOD)
        gg = pd.DataFrame({'t': sec[mm], 'r': reg[mm], 'l': lay[mm]}).groupby(['t', 'r', 'l']).indices; ii = np.flatnonzero(mm)
        kk = [k for k, v in gg.items() if len(v) >= 5]; ix = [ii[gg[k]] for k in kk]
        PB = np.array([np.log1p(1e4 * Yd[v].sum(0) / Ls[v].sum()) for v in ix]) / sdk
        return PB, np.array([len(v) for v in ix], float), *(np.array([k[j] for k in kk]) for j in range(3))

    odd = [s for s in SEC if int(s[1:]) % 2]; even = [s for s in SEC if not int(s[1:]) % 2]
    mu, ve = pb_fit(*blocks(SEC), lib, sec, SEC); mu = (mu - mu.mean()) / mu.std()
    mo, _ = pb_fit(*blocks(odd), lib, sec, SEC); me, _ = pb_fit(*blocks(even), lib, sec, SEC)
    sh = splithalf(mu, mo, me); lo = loo_splithalf(mo, me, mu)
    for k, (rd, sr) in enumerate(reading(mu, sh)):
        rows.append(dict(data=name, variant='pooled', comp=k + 1, var_share=ve[k], splithalf=sh[k], splithalf_loo=lo[k], SA_r=abs(sr),
                         depth_r=abs(np.corrcoef(mu[k], depth.reindex(mu.index))[0, 1]), reading=rd))
    del Y, Lg, Yd
    print(name, 'done', flush=True)

D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, 'resim_axes.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 45); print(D.round(3).to_string())
