"""Leiden NMI split into the real signal and the section effect.  python -m ottr.metrics.nmi_within <ds> <name,name,...>

Leiden exactly as ottr.metrics.evaluate (z-scored latent, 30 neighbours, resolution 1, seed 0); labels cached in
work/cache/leiden/<ds>__<name>.npy (one process per model: repeated Leiden calls in one process can segfault).
  within section : NMI(cluster, region) and NMI(cluster, layer) on the spots of ONE section, averaged over sections
                   -> how well clusters resolve regions / layers with no section effect possible
  within group   : NMI(cluster, section) on the spots of one group with >= 2 sections -> section effect (should be ~0)
  shared clusters: share of spots in clusters where no single section holds > 80% of the cluster
-> tables/nmi_within_<ds>.csv (rows of other models kept)."""
import os, sys, subprocess
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import normalized_mutual_info_score as NMI
from ..config import cfg, ensure

BAD = {'nan', 'remove', 'unassigned', 'white_matter'}


def leiden_file(ds, m):
    return os.path.join(ensure(os.path.join(cfg().cache, 'leiden')), f'{ds}__{m}.npy')


def one(ds, m):
    from .evaluate import leiden_labels
    F = np.load(cfg().emb(ds, m)).astype(float); F = F[:, F.std(0) > 0]; F = (F - F.mean(0)) / F.std(0)
    np.save(leiden_file(ds, m), leiden_labels(F))


def run(ds, models):
    c = cfg(); obs = ad.read_h5ad(c.inputs(ds), backed='r').obs
    sl = obs.slice.astype(str).values; grp = obs.group.astype(str).values; reg = obs.region.astype(str).values; lay = obs.layer.astype(str).values
    okr = ~np.isin(reg, list(BAD)); rows = []
    for m in models:
        if not os.path.exists(leiden_file(ds, m)): subprocess.run([sys.executable, '-m', 'ottr.metrics.nmi_within', ds, m, '--one'], check=True)
        L = np.load(leiden_file(ds, m))
        r = dict(model=m, n_clusters=len(set(L)), global_region=NMI(reg[okr], L[okr]), global_layer=NMI(lay, L), global_section=NMI(sl, L))
        r['within_section_region'] = np.mean([NMI(reg[(sl == s) & okr], L[(sl == s) & okr]) for s in np.unique(sl)])
        r['within_section_layer'] = np.mean([NMI(lay[sl == s], L[sl == s]) for s in np.unique(sl)])
        gs = [g for g in np.unique(grp) if len(np.unique(sl[grp == g])) >= 2]; ws, wr, sh = [], [], []
        for g in gs:
            mg = grp == g; ws.append(NMI(sl[mg], L[mg])); wr.append(NMI(reg[mg & okr], L[mg & okr]))
            t = pd.crosstab(L[mg], sl[mg]); dom = t.max(1) / t.sum(1); sh.append(t.sum(1)[dom <= 0.8].sum() / mg.sum())
            r[f'section_in_{g}'] = ws[-1]
        r['within_group_section'] = np.mean(ws); r['within_group_region'] = np.mean(wr); r['shared_cluster_spots'] = np.mean(sh)
        rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    R = pd.DataFrame(rows); out = os.path.join(ensure(c.tables()), f'nmi_within_{ds}.csv')
    if os.path.exists(out):
        old = pd.read_csv(out); R = pd.concat([old[~old.model.isin(R.model)], R], ignore_index=True)
    R.to_csv(out, index=False)
    return R


if __name__ == '__main__':
    if len(sys.argv) > 3 and sys.argv[3] == '--one': one(sys.argv[1], sys.argv[2])
    else: run(sys.argv[1], sys.argv[2].split(','))
