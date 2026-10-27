"""SimR, pooled pseudobulks: what drives the reproducible axis 1? Correlate region scores with the region's mean block size / block depth."""
import sys, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.argv = ['x', sys.argv[1]]
src = open(sys.argv[1] + '/pooled_test.py').read().split('rows, trows = [], []')[0]; exec(src)     # reuse pb_fit / splithalf / constants
ds = 'pfcserLSimR'; A = ad.read_h5ad(c.inputs(ds)); o = A.obs; Y = A.X.tocsr().astype(np.float32)
Ls = np.maximum(np.asarray(Y.sum(1)).ravel(), 1); sec = o.slice.astype(str).values; lay = o.layer.astype(str).values; reg = o.region.astype(str).values
lib = o.library_size.values.astype(float); SEC = sorted(set(sec), key=lambda t: int(t[1:]))
Lg = (sp.diags(1e4 / Ls) @ Y).tocsr(); Lg.data = np.log1p(Lg.data); sd = np.sqrt(np.asarray(Lg.multiply(Lg).mean(0)).ravel() - np.asarray(Lg.mean(0)).ravel() ** 2)
keep = sd > 0; Yk = Y[:, np.flatnonzero(keep)]; sd = sd[keep]
okl = np.isin(lay, LAYERS); okr = ~np.isin(reg, BAD); rc = pd.Series(reg[okl & okr]).value_counts(); GOOD = list(rc[rc >= 30].index)
mm = okl & okr & np.isin(reg, GOOD); gg = pd.DataFrame({'t': sec[mm], 'r': reg[mm], 'l': lay[mm]}).groupby(['t', 'r', 'l']).indices; ii = np.flatnonzero(mm)
kk = [k for k, v in gg.items() if len(v) >= 5]; ix = [ii[gg[k]] for k in kk]; W = np.array([len(v) for v in ix], float)
kt, kr, kl = (np.array([k[j] for k in kk]) for j in range(3))
PB = np.array([np.log1p(1e4 * np.asarray(Yk[v].sum(0)).ravel() / Ls[v].sum()) for v in ix]) / sd
mu, ve = pb_fit(PB, W, kt, kr, kl, lib, sec, SEC)
B = pd.DataFrame(dict(r=kr, l=kl, logn=np.log(W), logsumL=np.log([Ls[v].sum() for v in ix]), logmeanL=np.log([Ls[v].mean() for v in ix])))
for f in ('logn', 'logsumL', 'logmeanL'): B[f] -= B.groupby('l')[f].transform('mean')       # layer-centred, as the model
R = B.groupby('r')[['logn', 'logsumL', 'logmeanL']].mean().join(mu[[0, 1]])
print('n regions', len(R)); print(R.corr().loc[['logn', 'logsumL', 'logmeanL'], [0, 1]].round(2))
print('block spots: median', np.median(W), 'quantiles 10/90', np.percentile(W, [10, 90]))
m1 = ((mu[0] - mu[0].mean()) / mu[0].std()).sort_values(key=abs, ascending=False)
nsp = pd.Series(W).groupby(kr).sum(); nsec = pd.Series(kt).groupby(kr).nunique()
print(pd.DataFrame(dict(mu1_z=m1, spots=nsp.reindex(m1.index), sections=nsec.reindex(m1.index))).head(6).round(2))
odd = [s for s in SEC if int(s[1:]) % 2]; even = [s for s in SEC if not int(s[1:]) % 2]
def fit_half(keep_sec):
    q = np.isin(kt, keep_sec); return pb_fit(PB[q], W[q], kt[q], kr[q], kl[q], lib, sec, SEC)[0]
mo, me = fit_half(odd), fit_half(even)
for nm, h in (('odd', mo), ('even', me)):
    z = (h[0] - h[0].mean()) / h[0].std(); t = z.abs().sort_values(ascending=False); print(nm, 'axis1 top regions |z|:', t.head(3).round(2).to_dict())
top = m1.index[0]; cc = mo.index.intersection(me.index).drop(top, errors='ignore')
print('split-half |r| of axis 1 without region', top, ':', round(abs(np.corrcoef(mo.loc[cc, 0], me.loc[cc, 0])[0, 1]), 3))
