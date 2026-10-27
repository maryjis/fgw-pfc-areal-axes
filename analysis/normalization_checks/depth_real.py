import numpy as np, pandas as pd, anndata as ad, glob, os
R = '/workspace/Gene_modules/optimal_transport__tr_axeses'
o = ad.read_h5ad(f'{R}/work/inputs/pfcserL.h5ad', backed='r').obs
sec = o.slice.astype(str).values; lay = o.layer.astype(str).values; reg = o.region.astype(str).values
lt = np.log(o.library_size.values.astype(float)); ok = np.isin(lay, ['L1','L2','L3','L4','L5','L6'])
cs = pd.Series(lt).groupby(sec).transform(lambda v: np.log(np.median(np.exp(v)))).values
df = pd.DataFrame(dict(r=reg, l=lay, res=lt - cs))[ok]; df['res2'] = df.res - df.groupby('l').res.transform('mean')
rd = df.groupby('r').res2.mean()
C = pd.read_csv(f'{R}/work/inputs/pfcserL_columns.csv'); sa = C.groupby('area_level3').Ridge_SA_index.mean()
for f in sorted(glob.glob(f'{R}/work/axes/pfcserL*_*.npz')):
    if 'spot_axes' in f or 'Sim' in f: continue
    Z = np.load(f, allow_pickle=True)
    if 'pb_mu' not in Z: continue
    mu = pd.DataFrame(Z['pb_mu'], index=Z['pb_regions']); X = mu.join(rd.rename('depth')).join(sa.rename('SA'))
    print(os.path.basename(f), 'r(axis_k, region depth):', [round(X[k].corr(X.depth), 2) for k in range(5)], '| r(depth, SA)=', round(X.depth.corr(X.SA), 2), '| depth range x', np.round(np.exp(rd.reindex(mu.index)).agg(['min','max']).values, 2))
