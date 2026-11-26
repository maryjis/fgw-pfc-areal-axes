"""E6 of the validation plan: our gene sets in human snRNA-seq (Siletti et al. 2023), within supercluster, 8 PFC dissections vs our mu.
python -m ottr.validation.siletti
Per nucleus (all cortical dissections): the demo formula (ottr.genescore.pral.module_means on raw counts, all genes in the denominator,
symbols matched case-insensitively), z by the reference of snrna_types.reference per supercluster over all cortical dissections (equal weight per dissection),
score = z(+) - z(-); mean per dissection (and per donor x dissection). PFC dissections mapped to marmoset regions (MAP, approximate).
Units = 8 PFC dissections (exact permutation, 8! orders); spatial nulls = surrogates of mu carried through MAP. Superclusters with >= 50
nuclei in every PFC dissection. Primary: axes 1, 3 x {Upper-layer IT, Deep-layer IT}. Replication: per donor (>= 6 dissections).
-> work/validation/siletti_cells.parquet, results/tables/validation/siletti_scores.csv, rows_siletti.csv"""
import os, glob
import numpy as np
import pandas as pd
import anndata as ad
from multiprocessing import Pool
from ..config import cfg, ensure
from ..genescore import pral, snrna as SN
from . import common as C

DIR = '/workspace/external_repos/siletti2023'; MIN = 50
MAP = {'A24': ['A24a', 'A24b', 'A24c'], 'A32': ['A32', 'A32V'], 'A25': ['A25'], 'A14': ['A14R', 'A14C'], 'A13': ['A13a', 'A13b', 'A13L', 'A13M'],
       'FI': ['OPAl', 'OPro'], 'A44-A45': ['A45'], 'A46': ['A46D', 'A46V']}
PRIMARY = ['Upper-layer intratelencephalic', 'Deep-layer intratelencephalic']


def short(d):
    return d.split(' - ')[-1].strip()


def one(f):
    a = ad.read_h5ad(f); S = SN.gene_sets(); genes = a.var.feature_name.astype(str).str.upper().values; _, first = np.unique(genes, return_index=True); keep = np.sort(first)
    X = a.X[:, keep]; g = genes[keep]; out = a.obs[['donor_id', 'dissection', 'supercluster_term', 'total_UMIs']].copy(); out['dissection'] = out.dissection.astype(str).map(short)
    for s, sig in S.items():
        m, _ = pral.module_means(X, g, a.obs_names.values, {e: [x.upper() for x in v] for e, v in sig.items()}); out[f'{s}|Pr'] = m.Pr_signature_mean.values; out[f'{s}|Al'] = m.Al_signature_mean.values
    return out.reset_index(names='cell')


def cells():
    p = os.path.join(ensure(os.path.join(cfg().work, 'validation')), 'siletti_cells.parquet')
    if os.path.exists(p): return pd.read_parquet(p)
    fs = sorted(glob.glob(f'{DIR}/*.h5ad'))
    with Pool(8) as P: D = pd.concat(P.map(one, fs), ignore_index=True)
    for c in ('donor_id', 'dissection', 'supercluster_term'): D[c] = D[c].astype(str)
    D.to_parquet(p); return D


def scores(D):
    """z per supercluster with the sample-equal reference of snrna_types.reference (dissection = sample), score = z(+) - z(-)."""
    from ..genescore.snrna_types import reference
    out = []
    for sc, g in D.groupby('supercluster_term'):
        g = g[g.groupby('dissection').cell.transform('size') >= 2]; res = g[['cell', 'donor_id', 'dissection']].copy(); smp = g.dissection.values
        for s in SN.SCORES:
            v = 0
            for e, sg in (('Pr', 1), ('Al', -1)): m = g[f'{s}|{e}'].values.astype(float); c0, sd = reference(m, smp); v = v + sg * (m - c0) / sd
            res[s] = v
        res['supercluster'] = sc; out.append(res)
    return pd.concat(out, ignore_index=True)


def run():
    D = cells(); print('nuclei', len(D), 'dissections', D.dissection.nunique(), 'donors', sorted(D.donor_id.unique()), flush=True)
    Z = scores(D); A = Z.groupby(['supercluster', 'dissection'])[SN.SCORES].mean().join(Z.groupby(['supercluster', 'dissection']).size().rename('n')).reset_index()
    Ad = Z.groupby(['supercluster', 'donor_id', 'dissection'])[SN.SCORES].mean().join(Z.groupby(['supercluster', 'donor_id', 'dissection']).size().rename('n')).reset_index()
    pd.concat([A.assign(donor_id='all'), Ad]).to_csv(os.path.join(C.out_dir(), 'siletti_scores.csv'), index=False)
    pfc = list(MAP); rng = np.random.default_rng(37); rows = []
    for sc, g in A[A.dissection.isin(pfc)].groupby('supercluster'):
        g = g.set_index('dissection').reindex(pfc)
        if (g.n.fillna(0) < MIN).any(): continue
        for s in SN.SCORES:
            k = 1 if s == 'PrAl' else int(s[-1]); r = C.test(f'mu axis{k}', g[s].values, mapping=[MAP[d] for d in pfc], rng=rng, domain='human snRNA (Siletti 2023)', dataset=f'{sc}', species='human',
                                                             feature=f'{s} score in {sc}', family='primary' if (sc in PRIMARY and s in ('axis1', 'axis3')) else 'exploratory')
            rd = []
            for dn, h in Ad[(Ad.supercluster == sc) & Ad.dissection.isin(pfc) & (Ad.n >= 20)].groupby('donor_id'):
                h = h.set_index('dissection')
                if len(h) >= 6: x = C.carry(C.ours()[f'mu axis{k}'], [MAP[d] for d in h.index])[:, 0]; rd.append((dn, len(h), float(pd.Series(x).rank().corr(pd.Series(h[s].values).rank()))))
            r['replication'] = ('same' if all(np.sign(v) == np.sign(r['spearman']) for _, _, v in rd) else 'opposite') if rd else np.nan
            r['note'] = 'per donor: ' + '; '.join(f'{d} n={n} rho {v:+.2f}' for d, n, v in rd) + f'; nuclei per dissection {int(g.n.min())}-{int(g.n.max())}; FI and A44-A45 mapped approximately'
            rows.append(r)
    return C.save(rows, 'siletti')


if __name__ == '__main__':
    pd.set_option('display.width', 250); R = run()
    print(R[['dataset', 'ours', 'feature', 'spearman', 'p_perm', 'q_perm', 'p_msr', 'q_msr', 'p_vario', 'q_vario', 'replication', 'verdict']].round(4).to_string(index=False)); print(R[R.family == 'primary'][['feature', 'note']].to_string(index=False))
