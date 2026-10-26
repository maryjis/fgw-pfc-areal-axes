"""All metrics of the serial-series benchmark and the model ranking (step 06).
python -m ottr.metrics.series "<label>:<embedding>,..." [--out tables/series_metrics.csv]

Per embedding (z-scored, constant dimensions dropped):
  from evaluate (cache):  mixing_all (cross-section kNN share / chance), area_transfer (column profiles, held-out section),
                          layer_heldout, SA (best |Spearman| column PC1-5 vs SA index)
  from nmi_within:        section_nmi (Leiden NMI with the section inside the group), region / layer NMI inside a section
  neighbour sections:     for every section, 2 000 spots and the spots of the sections +-1:
                          mixing_nb (share of 30-NN from the neighbours / their share of the pool), layer_nb, region_nb and
                          area_nb (15-NN majority within the same layer, neighbour sections only)
  variance:               eta2 of region / coarse area / section within each of L2, L3, L5, L6 (correlation ratio of the
                          latent, averaged over layers)
  k-means:                NMI of k-means (7) clusters with the coarse area and with the section, within L3 and L5
Ranking (rank()): score = mean of min-max scaled mixing, region in layer (region NMI in section), area transfer,
layer held-out, SA; place by score."""
import os, json, argparse
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from ..config import cfg, ensure
from ..utils.labels import AREA
from .evaluate import cache_path

RANK_METRICS = {'mixing_all': 'mixing', 'region_nmi_in_section': 'region_in_layer', 'area_transfer': 'area_transfer', 'layer_heldout': 'layer', 'SA': 'SA'}


def eta(F, g):
    """Correlation ratio: share of the total variance of F explained by the group means."""
    mu = F.mean(0); m = pd.DataFrame(F).groupby(g).transform('mean').values; return ((m - mu) ** 2).sum() / ((F - mu) ** 2).sum()


def mode(a):
    u, c = np.unique(a, return_inverse=True); c = c.reshape(a.shape); cnt = np.zeros((a.shape[0], len(u)), np.int32)
    np.add.at(cnt, (np.repeat(np.arange(a.shape[0]), a.shape[1]), c.ravel()), 1); return u[cnt.argmax(1)]


def series_metrics(items, ds=None, nmi_file=None):
    c = cfg(); ds = ds or c.series['name']
    o = ad.read_h5ad(c.inputs(ds), backed='r').obs; sl = o.slice.astype(str).values; lay = o.layer.astype(str).values; reg = o.region.astype(str).values
    RG = ~np.isin(reg, ['nan', 'None', 'remove', 'unassigned']); area = np.array([AREA.get(r, 'nan') for r in reg]); AO = area != 'nan'
    SEC = sorted(set(sl), key=lambda t: int(t[1:])); si = np.array([SEC.index(s) for s in sl]); L4 = ['L2', 'L3', 'L5', 'L6']
    nw = pd.read_csv(nmi_file or c.tables(f'nmi_within_{ds}.csv')).set_index('model')
    rows = []
    for lab, e in items:
        m = json.load(open(cache_path(ds, e)))['metrics']; w = nw.loc[e]
        F = np.load(c.emb(ds, e)).astype(np.float32); F = F[:, F.std(0) > 0]; F = (F - F.mean(0)) / F.std(0)
        rng = np.random.default_rng(0); mix, racc, lacc, aacc = [], [], [], []
        for i in range(len(SEC)):
            nb = np.abs(si - i) == 1; me = si == i; pool = np.flatnonzero(me | nb); q = rng.choice(np.flatnonzero(me), min(2000, me.sum()), replace=False)
            _, j = NearestNeighbors(n_neighbors=30, n_jobs=16).fit(F[pool]).kneighbors(F[q]); mix.append(nb[pool[j]].mean() / nb[pool].mean())
            a = q[np.isin(lay[q], ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])]; b = np.flatnonzero(nb & np.isin(lay, ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']))
            if len(a) and len(b): _, j = NearestNeighbors(n_neighbors=15, n_jobs=16).fit(F[b]).kneighbors(F[a]); lacc.append(np.mean(mode(lay[b][j]) == lay[a]))
            for l in L4:
                a = np.flatnonzero(me & (lay == l) & RG); b = np.flatnonzero(nb & (lay == l) & RG)
                if len(a) < 50 or len(b) < 50: continue
                a = rng.choice(a, min(800, len(a)), replace=False); _, j = NearestNeighbors(n_neighbors=15, n_jobs=16).fit(F[b]).kneighbors(F[a])
                racc.append(np.mean(mode(reg[b][j]) == reg[a])); aacc.append(np.mean(mode(area[b][j]) == area[a]))
        r = dict(model=lab, emb=e, mixing_all=m['align: cross-slice share'] / m['align: cross-slice share, chance'], section_nmi=w['within_group_section'],
                 area_transfer=m.get('region: column area transfer, slice z-scored'), layer_heldout=m['layer: held-out slice (spot kNN)'],
                 SA=m['SA: best |Spearman| PC1-5 vs SA index'], region_nmi_in_section=w['within_section_region'], layer_nmi_in_section=w['within_section_layer'],
                 mixing_nb=np.mean(mix), region_nb=np.mean(racc), area_nb=np.mean(aacc), layer_nb=np.mean(lacc),
                 eta2_region=np.mean([eta(F[(lay == l) & RG], reg[(lay == l) & RG]) for l in L4]),
                 eta2_area=np.mean([eta(F[(lay == l) & AO], area[(lay == l) & AO]) for l in L4]),
                 eta2_section=np.mean([eta(F[(lay == l) & RG], sl[(lay == l) & RG]) for l in L4]))
        for l in ('L3', 'L5'):
            q = (lay == l) & AO; k = KMeans(7, n_init=4, random_state=0).fit_predict(F[q]); r[f'nmi_area_{l}'] = nmi(area[q], k); r[f'nmi_section_{l}'] = nmi(sl[q], k)
        rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return pd.DataFrame(rows)


def rank(g):
    """score = mean of min-max scaled ranking metrics; place by score; mean rank over the same metrics."""
    g = g.set_index('model')
    Z = pd.DataFrame({m: (g[m] - g[m].min()) / (g[m].max() - g[m].min()) for m in RANK_METRICS}); rk = pd.DataFrame({m: g[m].rank(ascending=False) for m in RANK_METRICS})
    T = g[list(RANK_METRICS) + ['section_nmi']].rename(columns=RANK_METRICS).assign(score=Z.mean(1), mean_rank=rk.mean(1))
    T['place'] = T.score.rank(ascending=False).astype(int)
    return T.sort_values('place')


if __name__ == '__main__':
    P = argparse.ArgumentParser(); P.add_argument('items'); P.add_argument('--out', default=None); P.add_argument('--rank_out', default=None)
    a = P.parse_args(); items = [tuple(x.split(':')) for x in a.items.split(',')]
    R = series_metrics(items); out = a.out or cfg().tables('series_metrics.csv'); ensure(os.path.dirname(out)); R.to_csv(out, index=False)
    T = rank(R); T.to_csv(a.rank_out or cfg().tables('series_rank.csv')); print(T.round(3).to_string())
