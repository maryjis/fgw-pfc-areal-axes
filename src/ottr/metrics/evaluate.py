"""Global metrics of an embedding (one json per embedding in the cache).  python -m ottr.metrics.evaluate <ds> <name> [--leiden]

The embedding is z-scored per dimension (constant dimensions dropped) before every metric.
  alignment : share of the 30 nearest neighbours that come from another section, divided by its chance level (1 - sum p_s^2)
              -> 'mixing'; slice entropy among the neighbours (iLISI-like, 0..1)
  region    : same region among cross-section neighbours; region of a spot from 15-NN of the OTHER sections within the same
              layer (balanced accuracy, held-out section); area transfer: kNN (k 5) of column profiles, train on the other
              sections, predict the coarse area of the held-out section (column profiles z-scored within section)
  layer     : layer of a spot from 15-NN of the other sections (balanced accuracy, held-out section)
  SA        : best |Spearman| of column PC1-5 with the Ridge SA index; ridge from the column profile to the SA index on a
              held-out section
  extra     : Leiden clusters (30-NN graph, resolution 1, seed 0) and their NMI with region, layer, section (--leiden)
Logic copied unchanged from the project's evaluate.py."""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, normalized_mutual_info_score as NMI
from sklearn.linear_model import RidgeCV
from ..config import cfg, ensure

BAD = {'remove', 'white_matter', 'nan', 'WM', 'None'}
LAYERS = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']


class Context:
    """Labels, columns and column blocks of a dataset (computed once, reused for every embedding)."""

    def __init__(self, ds):
        c = cfg(); self.ds = ds
        obs = ad.read_h5ad(c.inputs(ds), backed='r').obs.copy(); self.n = len(obs)
        CM = pd.read_csv(c.columns(ds))
        self.sl = obs.slice.astype(str).values; self.grp = obs.group.astype(str).values
        self.reg = obs.region.astype(str).values; self.lay = obs.layer.astype(str).values
        self.okreg = ~np.isin(self.reg, list(BAD)); self.slices = list(dict.fromkeys(self.sl))
        self.same_group_other = 'same group' if (ds == 'adult5' or ds.startswith('pfc')) else 'same structure, other animal'
        REQ = ['L2', 'L3', 'L5'] if ds.startswith(('pfc405', 'pfcser')) else ['L2', 'L3', 'L5', 'L6']
        self.MSAA = [s for s in self.slices if ('MSAA' in s) or (ds == 'adult5' and s in ('T442', 'T454', 'T457'))]
        col = obs['col'].values.astype(int)
        gi = pd.DataFrame({'slice': self.sl, 'col': col, 'layer': self.lay})[col >= 0].groupby(['slice', 'col', 'layer']).indices
        gi = {k: np.flatnonzero(col >= 0)[v] for k, v in gi.items()}
        KEYS, BLOCKS = [], []
        for s, cc in sorted({(s, cc) for s, cc, _ in gi}):
            bl = {l: gi.get((s, cc, l)) for l in ['L2', 'L3', 'L4', 'L5', 'L6']}
            if any(bl[l] is None or len(bl[l]) < 3 for l in REQ): continue
            KEYS.append((s, cc)); BLOCKS.append([b for b in bl.values() if b is not None and len(b) >= 3])
        self.K = pd.DataFrame(KEYS, columns=['slice', 'col']).merge(CM, on=['slice', 'col'], how='left'); self.BLOCKS = BLOCKS
        loglib = np.log(obs.library_size.values.astype(float))
        self.LOGLIB_COL = np.array([np.mean([loglib[b].mean() for b in bl]) for bl in BLOCKS])

    def col_profiles(self, F):
        X = np.array([np.mean([F[b].mean(0) for b in bl], axis=0) for bl in self.BLOCKS])
        X = X[:, X.std(0) > 0]; X = (X - X.mean(0)) / X.std(0); Xc = X.copy()
        for s in self.K.slice.unique():
            m = (self.K.slice == s).values; Xc[m] = np.clip((Xc[m] - Xc[m].mean(0)) / (Xc[m].std(0) + 1e-9), -10, 10)
        return X, Xc

    def knn_heldout(self, F, labels, mask, by_layer):
        accs = []
        for l in (LAYERS if by_layer else [None]):
            lm = mask & ((self.lay == l) if l else True)
            for t in self.slices:
                tr = lm & (self.sl != t); te = lm & (self.sl == t)
                com = list(set(labels[tr]) & set(labels[te]))
                if len(com) < 2: continue
                tr &= np.isin(labels, com); te &= np.isin(labels, com)
                if te.sum() < 50: continue
                p = KNeighborsClassifier(15, n_jobs=32).fit(F[tr], labels[tr]).predict(F[te])
                accs.append(balanced_accuracy_score(labels[te], p))
        return float(np.mean(accs))

    def col_transfer(self, X):
        K = self.K; acc = []
        for t in K.slice.unique():
            tr = (K.slice != t).values; te = (K.slice == t).values
            com = set(K.area2[tr]) & set(K.area2[te]); tr &= K.area2.isin(com).values; te &= K.area2.isin(com).values
            if te.sum() < 10 or len(com) < 3: continue
            acc.append(balanced_accuracy_score(K.area2[te], KNeighborsClassifier(5).fit(X[tr], K.area2[tr]).predict(X[te])))
        return float(np.mean(acc))

    def evaluate(self, F, leiden=False):
        n, sl, grp, reg, lay, okreg, slices, K = self.n, self.sl, self.grp, self.reg, self.lay, self.okreg, self.slices, self.K
        rng = np.random.default_rng(0)
        F = np.asarray(F, float); F = F[:, F.std(0) > 0]; F = (F - F.mean(0)) / F.std(0)
        r = {}
        _, idx = NearestNeighbors(n_neighbors=31, n_jobs=32).fit(F).kneighbors(F); idx = idx[:, 1:]
        rows = np.repeat(np.arange(n), 30); cols = idx.ravel(); x = sl[rows] != sl[cols]
        p = pd.Series(sl).value_counts(normalize=True)
        r['align: cross-slice share'] = x.mean(); r['align: cross-slice share, chance'] = 1 - (p ** 2).sum()
        codes = pd.Categorical(sl, categories=slices).codes; cnt = np.zeros((n, len(slices)))
        np.add.at(cnt, (np.repeat(np.arange(n), 30), codes[cols]), 1); q = cnt / 30
        r['align: slice entropy (iLISI-like)'] = float(np.mean(-(q * np.log(q + 1e-12)).sum(1)) / np.log(len(slices)))
        r[f'align: neighbours from other slice, {self.same_group_other}'] = (x & (grp[rows] == grp[cols])).mean()
        xr = x & okreg[rows] & okreg[cols]
        r['region: same region among cross-slice neighbours'] = (reg[rows][xr] == reg[cols][xr]).mean()
        null = []
        for _ in range(3):
            perm = reg.copy()
            for s in slices: m = sl == s; perm[m] = rng.permutation(perm[m])
            null.append((perm[rows][xr] == perm[cols][xr]).mean())
        r['region: same region among cross-slice neighbours, chance'] = float(np.mean(null))
        r['layer: same layer among cross-slice neighbours'] = (lay[rows][x] == lay[cols][x]).mean()
        r['region: within layer, held-out slice (spot kNN)'] = self.knn_heldout(F, reg, okreg, True)
        r['layer: held-out slice (spot kNN)'] = self.knn_heldout(F, lay, np.isin(lay, LAYERS), False)
        X, Xc = self.col_profiles(F)
        r['region: column area transfer'] = self.col_transfer(X); r['region: column area transfer, slice z-scored'] = self.col_transfer(Xc)
        m = K.Ridge_SA_index.notna().values; rho = []
        for t in K.slice.unique():
            te = m & (K.slice == t).values; tr = m & (K.slice != t).values
            if te.sum() < 10: continue
            mdl = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], K.Ridge_SA_index[tr])
            rho.append(spearmanr(mdl.predict(X[te]), K.Ridge_SA_index[te])[0])
        r['SA: ridge on held-out slice (Spearman)'] = float(np.mean(rho))
        U, S, _ = np.linalg.svd(Xc - Xc.mean(0), full_matrices=False); PC = U[:, :10] * S[:10]
        r['SA: best |Spearman| PC1-5 vs SA index'] = max(abs(spearmanr(PC[m, k], K.Ridge_SA_index[m])[0]) for k in range(min(5, PC.shape[1])))
        ms = K.slice.isin(self.MSAA).values & K.SA.isin(['S', 'A']).values
        if ms.sum() > 20 and K.SA[ms].nunique() == 2:
            y = (K.SA[ms] == 'S').values
            r['SA: S/A AUROC on MSAA columns, best PC1-10'] = max(max(roc_auc_score(y, PC[ms, k]), 1 - roc_auc_score(y, PC[ms, k])) for k in range(PC.shape[1]))
        r['tech: |rho| column PC1 vs log library size'] = abs(spearmanr(PC[:, 0], self.LOGLIB_COL)[0])
        if leiden:
            L = leiden_labels(F)
            r['extra: Leiden NMI region'] = NMI(reg[okreg], L[okreg]); r['extra: Leiden NMI layer'] = NMI(lay, L); r['extra: Leiden NMI slice'] = NMI(sl, L)
            r['extra: Leiden n clusters'] = int(len(np.unique(L)))
        return {k: float(v) for k, v in r.items()}


def leiden_labels(F):
    """Leiden clusters of a z-scored latent: 30-NN graph (scanpy), resolution 1.0, seed 0, igraph flavour, 2 iterations."""
    import scanpy as sc
    b = ad.AnnData(np.zeros((len(F), 1), np.float32)); b.obsm['rep'] = F.astype(np.float32)
    sc.pp.neighbors(b, use_rep='rep', n_neighbors=30, random_state=0)
    sc.tl.leiden(b, resolution=1.0, random_state=0, flavor='igraph', n_iterations=2)
    return b.obs.leiden.values.astype(int)


def cache_path(ds, name):
    return os.path.join(ensure(cfg().cache), f'{ds}__{name}__seed0.json')


def run(ds, names, leiden=True, force=False):
    """Evaluate embeddings work/embeddings/<ds>/<name>__seed0.npy; one json per embedding in work/cache/."""
    ctx = None
    for name in names:
        out = cache_path(ds, name)
        if os.path.exists(out) and not force: continue
        ctx = ctx or Context(ds)
        Z = np.load(cfg().emb(ds, name)); assert len(Z) == ctx.n, (name, Z.shape, ctx.n)
        t0 = time.time(); r = ctx.evaluate(Z, leiden)
        json.dump(dict(dataset=ds, method=name, seed=0, dim=int(Z.shape[1]), metrics=r), open(out, 'w'), indent=1)
        print(f'{name}: {time.time() - t0:.0f}s', {k: round(v, 3) for k, v in r.items()}, flush=True)


if __name__ == '__main__':
    P = argparse.ArgumentParser(); P.add_argument('ds'); P.add_argument('names'); P.add_argument('--leiden', action='store_true'); P.add_argument('--force', action='store_true')
    a = P.parse_args(); run(a.ds, a.names.split(','), a.leiden, a.force)
