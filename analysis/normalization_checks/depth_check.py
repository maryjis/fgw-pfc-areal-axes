import numpy as np, pandas as pd, anndata as ad, sys
sys.path.insert(0,'src')
R = '/workspace/Gene_modules/optimal_transport__tr_axeses'
A = ad.read_h5ad(f'{R}/work/inputs/pfcserLSimR.h5ad', backed='r'); o = A.obs
sec = o.slice.astype(str).values; lay = o.layer.astype(str).values; reg = o.region.astype(str).values
tot = np.asarray(A.X[:].sum(1)).ravel()                           # per-spot UMI in the simulated data (= real totals)
lt = np.log(np.maximum(tot, 1))
LAY = ['L1','L2','L3','L4','L5','L6']; ok = np.isin(lay, LAY)
# what the model removes: log median UMI per SECTION
cs = pd.Series(lt).groupby(sec).transform(lambda v: np.log(np.median(np.exp(v)))).values
df = pd.DataFrame(dict(r=reg, l=lay, s=sec, lt=lt, res=lt - cs))[ok]
# region depth after removing section level and layer level (as the model: centre per layer)
df['res2'] = df.res - df.groupby('l').res.transform('mean')
rd = df.groupby('r').agg(n=('lt','size'), logUMI=('lt','mean'), resid=('res2','mean'))
Z = np.load(f'{R}/work/axes/pfcserLSimR_simR_Expression.npz', allow_pickle=True)
mu = pd.DataFrame(Z['pb_mu'], index=Z['pb_regions'])
C = pd.read_csv(f'{R}/work/inputs/pfcserL_columns.csv'); sa = C.groupby('area_level3').Ridge_SA_index.mean()
X = rd.join(mu[[0,1]].rename(columns={0:'axis1',1:'axis2'}), how='inner').join(sa.rename('SA'))
print('regions', len(X)); print('UMI range across regions (exp of resid):', np.round(np.exp(X.resid).agg(['min','max']).values, 2))
print(X[['resid','axis1','axis2','SA']].corr().round(2))
# gene loadings vs gene mean (does the axis follow the gene mean?)
G = Z['pb_G'][0]; hv = Z['pb_has_layer'].any(0); d = G[hv].sum(0)
mean_g = np.asarray(A.X[::5].mean(0)).ravel()[Z['keepf']] if 'keepf' in Z else None
from scipy.stats import spearmanr
print('axis1 gene loading vs log gene mean, Spearman:', round(spearmanr(d, np.log(mean_g + 1e-9))[0], 2))
