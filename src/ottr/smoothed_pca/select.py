"""Choice of the smoothing radius of the smoothed PCA (step 02).  python -m ottr.smoothed_pca.select [--ds pfcserL] [--radii ...]

Why a rule is needed: a larger disk removes more noise and mixes the sections better, but blurs the layers (a disk wider than a
layer averages neighbouring layers) and finally the areas. The anchor space of FGW must keep the layers, because the 2D
structure term matches nodes by depth and the 1D profiles are layer means.
Rule: evaluate the 'centred' curve embedding of every radius (ottr.metrics.evaluate) and take the SMALLEST radius whose layer
transfer to a held-out section ('layer: held-out slice (spot kNN)') is within `tol` (0.005) of the best value on the curve -
the least smoothing that already gives the full layer signal. The other metrics (mixing, region within layer, area transfer,
SA) and a combined score (mean of min-max scaled metrics) are reported next to it.
On the adult5 and P0 benchmarks this picks r = 1.5 grid steps (75 um, ~9 spots), the radius used throughout.
-> tables/smoothing_radius_<ds>.csv"""
import os, argparse, subprocess, sys, json
import numpy as np
import pandas as pd
from ..config import cfg, ensure
from .curve import curve
from ..metrics.evaluate import cache_path

METRICS = {'layer': 'layer: held-out slice (spot kNN)', 'mixing': 'align: cross-slice share', 'region_in_layer': 'region: within layer, held-out slice (spot kNN)',
           'area_transfer': 'region: column area transfer, slice z-scored', 'SA': 'SA: best |Spearman| PC1-5 vs SA index'}


def choose(table, tol=0.005, key='layer'):
    """table: one row per radius with the metric columns. Returns the chosen radius."""
    t = table.sort_values('r'); best = t[key].max()
    return float(t[t[key] >= best - tol].r.min())


def table_from_cache(ds, radii):
    rows = []
    for r in radii:
        m = json.load(open(cache_path(ds, f'curve_r{r:g}_centred')))['metrics']
        rows.append(dict(r=r, **{k: m[v] for k, v in METRICS.items()}))
    T = pd.DataFrame(rows); M = list(METRICS)
    T['score'] = ((T[M] - T[M].min()) / (T[M].max() - T[M].min())).mean(1)
    return T


def run(ds, radii, tol=0.005):
    c = cfg()
    missing = [r for r in radii if not os.path.exists(c.emb(ds, f'curve_r{r:g}_centred'))]
    if missing: curve(ds, missing, variants=('none', 'centred'))
    for r in radii:                                    # one process per embedding (as evaluate.py)
        if not os.path.exists(cache_path(ds, f'curve_r{r:g}_centred')):
            subprocess.run([sys.executable, '-m', 'ottr.metrics.evaluate', ds, f'curve_r{r:g}_centred'], check=True)
    T = table_from_cache(ds, radii); r0 = choose(T, tol); T['chosen'] = T.r == r0
    T.to_csv(os.path.join(ensure(c.tables()), f'smoothing_radius_{ds}.csv'), index=False)
    print(T.round(3).to_string(index=False)); print('chosen radius', r0)
    return r0


if __name__ == '__main__':
    P = argparse.ArgumentParser(); P.add_argument('--ds', default=cfg().series['name']); P.add_argument('--radii', default='0,1.5,2.5,3.5,5,7')
    P.add_argument('--tol', type=float, default=0.005); a = P.parse_args()
    run(a.ds, [float(x) for x in a.radii.split(',')], a.tol)
