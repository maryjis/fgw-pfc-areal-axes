"""Simulation test of the areal-axis model on the real geometry of the series (optional step 09b).
python -m ottr.areal.simulation make | eval

make: two simulated data sets with the spots, coordinates, layers, regions, columns and depth of the series; only the counts
are replaced (genes = the 1 937 model genes; per-spot total = the spot's real total, so sequencing depth varies as in the data):
  <series>SimR (random)           every spot draws from ONE gene distribution (multinomial with the real mean proportions) -
                                  no layers, no regions, no sections
  <series>SimK (three components) log rate = b_g + L_layer,g + s_section,g + sum_m a_m mu_m(region) w_m,g with mu_1 = region SA
                                  index, mu_2, mu_3 random (orthogonal, unit variance), amplitudes 0.5, 0.35, 0.25
  writes inputs, runs/input (per-section files, for an optional FGW training on the simulated data) and the 'expression'
  embedding expr_logcp10k (log1p CP10k / gene SD); truth in tables/sim/truth_{regions,genes}.csv
eval: the areal model on the expression (multi_axes, tags _simR / _simK) and, if present, on an FGW latent of the simulated
  data (embedding 'fgw2d', tag _simRf / _simKf): per recovered axis |r| with every true component over regions; R2 of each true
  component from the five recovered axes (adjusted) with the 95% null of random region vectors -> tables/sim/sim_eval_*.csv
Finding of the original test: on random counts log1p(CP10k) + region-level differences of sequencing depth produce a false
SA-like axis (gene loadings follow the gene mean), in the expression and in the FGW latent alike."""
import os, sys, shutil, json
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from ..config import cfg, ensure
from . import multi_axes

SETS = [('SimR', '_simR', 'random'), ('SimK', '_simK', 'three components')]


def make(seed=2026):
    c = cfg(); base = c.series['name']; rng = np.random.default_rng(seed); OUT = ensure(c.tables('sim'))
    A = ad.read_h5ad(c.inputs(base)); o = A.obs.copy(); genes = np.array(A.var_names); G = len(genes)
    X = A.X.tocsr().astype(np.float64); tot = np.asarray(X.sum(1)).ravel().astype(int); tot[tot < 1] = 1
    p0 = np.asarray(X.sum(0)).ravel() + 1; p0 /= p0.sum(); b = np.log(p0)
    sec = o.slice.astype(str).values; lay = o.layer.astype(str).values; reg = o.region.astype(str).values
    C = pd.read_csv(c.columns(base)); sa = C[C.Ridge_SA_index.notna()].groupby('area_level3').Ridge_SA_index.mean()
    REG = sorted(set(reg) - {'nan', 'None', 'remove', 'unassigned'})
    M = np.zeros((len(REG), 3)); M[:, 0] = sa.reindex(REG).fillna(sa.mean()).values; M[:, 1:] = rng.standard_normal((len(REG), 2))
    M -= M.mean(0); Q, _ = np.linalg.qr(M); M = Q * np.sqrt(len(REG)) * np.sign(np.diag(Q.T @ M))[None, :]
    TR = pd.DataFrame(M, index=REG, columns=['mu1_SA', 'mu2_random', 'mu3_random']); TR.index.name = 'region'; TR.to_csv(f'{OUT}/truth_regions.csv')
    W = rng.standard_normal((3, G)); AMP = np.array([0.5, 0.35, 0.25])
    pd.DataFrame(W.T, index=genes, columns=['w1', 'w2', 'w3']).rename_axis('gene').to_csv(f'{OUT}/truth_genes.csv')
    LAYS = sorted(set(lay)); Lp = {l: rng.normal(0, 0.5, G) for l in LAYS}; Sp = {s: rng.normal(0, 0.15, G) for s in sorted(set(sec))}
    mu_spot = np.zeros((len(o), 3)); idx = {r: i for i, r in enumerate(REG)}
    for i, r in enumerate(reg):
        if r in idx: mu_spot[i] = M[idx[r]]

    def draw(logit_rows, n):
        p = np.exp(logit_rows - logit_rows.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        return np.vstack([rng.multinomial(t, pp) for t, pp in zip(n, p)]).astype(np.float32)

    for suf, _, _ in SETS:
        ds = base + suf; rows = []
        for s0 in range(0, len(o), 20000):
            sl = slice(s0, min(len(o), s0 + 20000))
            if suf == 'SimR': lg = np.repeat(b[None, :], sl.stop - sl.start, 0)
            else: lg = b[None, :] + np.vstack([Lp[l] for l in lay[sl]]) + np.vstack([Sp[s] for s in sec[sl]]) + (mu_spot[sl] * AMP[None, :]) @ W
            rows.append(sp.csr_matrix(draw(lg, tot[sl])))
        Y = sp.vstack(rows).tocsr(); print(ds, 'counts', Y.shape, flush=True)
        S = ad.AnnData(Y, obs=o.copy(), var=pd.DataFrame(index=genes)); S.obsm['spatial'] = np.asarray(A.obsm['spatial']); S.write_h5ad(c.inputs(ds), compression='gzip')
        shutil.copy(c.columns(base), c.columns(ds)); d = ensure(c.run_input(ds))
        for full in dict.fromkeys(o.slice_full.astype(str).values):
            q = (o.slice_full.astype(str).values == full); bc = [x.rsplit('-', 1)[0] for x in o.index[q].astype(str)]
            R_ = ad.AnnData(Y[np.flatnonzero(q)], obs=pd.DataFrame(index=bc), var=pd.DataFrame(index=genes)); R_.obsm['spatial'] = np.asarray(A.obsm['spatial'])[q]
            R_.write_h5ad(f'{d}/{full}_b100_filtered.h5ad')
        L = (sp.diags(1e4 / np.maximum(np.asarray(Y.sum(1)).ravel(), 1)) @ Y).tocsr(); L.data = np.log1p(L.data); L = L.toarray().astype(np.float32)
        sd = L.std(0); keep = sd > 0; ensure(c.emb_dir(ds)); np.save(c.emb(ds, 'expr_logcp10k'), L[:, keep] / sd[keep]); print(ds, 'written', flush=True)


def adj_r2(y, X):
    X1 = np.c_[np.ones(len(y)), X]; b, *_ = np.linalg.lstsq(X1, y, rcond=None); r2 = 1 - ((y - X1 @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    n, p = X.shape; return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def evaluate():
    c = cfg(); base = c.series['name']; rng = np.random.default_rng(0); OUT = ensure(c.tables('sim'))
    TR = pd.read_csv(f'{OUT}/truth_regions.csv', index_col=0); TRUE = list(TR.columns); rows, trows = [], []
    for suf, tag, lbl in SETS:
        ds = base + suf; o = ad.read_h5ad(c.inputs(ds), backed='r').obs; reg = o.region.astype(str).values; sec = o.slice.astype(str).values; col = o.col.values.astype(int)
        runs = [('Expression', tag, 'expr_logcp10k')] + ([('FGW 2D', tag + 'f', 'fgw2d')] if os.path.exists(c.emb(ds, 'fgw2d')) else [])
        for m, tg, emb in runs:
            Cp = multi_axes.run([(m, emb)], ds=ds, tag=tg)
            Z = np.load(os.path.join(multi_axes.axes_dir(), f'{ds}{tg}_{m.replace(" ", "_")}.npz'), allow_pickle=True)
            for an in ('pseudobulk', 'column'):
                if an == 'pseudobulk': R_ = pd.DataFrame(Z['pb_mu'][:, :5], index=Z['pb_regions'])
                else:
                    d = dict(zip(Z['col_keys'], Z['col_scores'][:, :5].astype(float))); V = np.array([d.get(f'{s}|{cc}', [np.nan] * 5) for s, cc in zip(sec, col)], float)
                    q = np.isfinite(V[:, 0]) & np.isin(reg, TR.index); R_ = pd.DataFrame(V[q]).groupby(reg[q]).mean()
                R_ = R_.dropna(); ix = R_.index.intersection(TR.index); R_, T_ = R_.loc[ix], TR.loc[ix]; t = Cp[(Cp.model == m) & (Cp.analysis == an)].set_index('comp')
                for k in range(R_.shape[1]):
                    r = {tc: abs(np.corrcoef(R_[k], T_[tc])[0, 1]) for tc in TRUE}
                    rows.append(dict(data=lbl, model=m, analysis=an, comp=k + 1, var_share=t.loc[k + 1].var_share, reading=t.loc[k + 1].reading,
                                     splithalf=t.loc[k + 1].get('splithalf', np.nan), **{f'abs_r_{tc}': r[tc] for tc in TRUE}, best_true=max(r, key=r.get), best_abs_r=max(r.values())))
                null = np.percentile([adj_r2(rng.standard_normal(len(ix)), R_.values) for _ in range(200)], 95)
                for tc in TRUE: trows.append(dict(data=lbl, model=m, analysis=an, truth=tc, adjR2_from_5_axes=adj_r2(T_[tc].values, R_.values), null95=null, n_regions=len(ix)))
            print(ds, m, 'done', flush=True)
    A_ = pd.DataFrame(rows); A_.to_csv(f'{OUT}/sim_eval_axes.csv', index=False); B_ = pd.DataFrame(trows); B_.to_csv(f'{OUT}/sim_eval_truth.csv', index=False)
    return A_, B_


if __name__ == '__main__':
    {'make': make, 'eval': evaluate}[sys.argv[1]]()
