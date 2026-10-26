"""Main axes of every model's latent by four methods, the L2/L3 analysis and the cross-method overlap (step 07).
python -m ottr.areal.multi_axes "<label>:<embedding>,..." [--ds pfcserL] [--tag _x]

  spot       : PCA of the latent on all L1-L6 spots (5 PCs)
  column     : PCA of column profiles (mean of the L2-L6 layer-block means of a column; >= 3 layers with >= 3 spots)
  pseudobulk : the AREAL-AXIS MODEL (the main analysis), K = 5 axes, needs >= 6 regions:
                 x_trl = f_l + theta c_t + sum_m mu_m(r) g_ml + e
               x_trl = mean latent (each dimension divided by its SD) of the spots of section t, region r, layer l (>= 5 spots)
               f_l   = layer profile (removed: blocks centred within each layer, weighted by the number of spots)
               c_t   = sequencing depth of the section (log median library, centred); theta its loading (removed by
                       weighted least squares)
               mu_m(r) = score of region r on areal axis m, shared by all layers; g_ml = loading of axis m in layer l.
               Fit: average the residual blocks per (region, layer), concatenate the layers of a region into one row
               (regions with >= 3 layers), centre, SVD -> mu (left vectors x sqrt(n regions)) and g (right vectors).
               So an areal axis is a direction that orders the REGIONS consistently across layers and sections;
               variance share = share of the between-region variance; split-half = |r| of mu between the fits on the
               odd and on the even sections (reproducibility).
  L2, L3     : PCA on the spots of one layer; shared = principal vectors of the L2 and L3 5-PC subspaces
Readings (one rule set) name every axis: section effect > depth > SA axis (|Spearman| with the column SA index >= 0.6) >
region axis > sequencing depth > gene programme named by marker genes at both ends (correlation with log CP10k smoothed over
r 1.5 and centred per section). Pseudobulk axes: SA (|r| with the region SA index >= 0.6) > arch of axis 1 (parabola R2 >= 0.5)
> rostro-caudal (|r| with the region's mean section number >= 0.6) > not reproducible (split-half < 0.5) > areal axis.
Outputs: tables/axes_<ds><tag>_comp.csv (one row per axis), work/axes/<ds><tag>_<model>.npz (scores, directions, loadings),
tables/axes_<ds><tag>_cross.csv (|cos| between the methods' directions). Logic copied unchanged from multi_axes.py."""
import os, argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.stats import spearmanr
from ..config import cfg, ensure
from ..utils.labels import MARK, KNOWN, genes_tag
from ..utils.smoothing import grid_positions, disk_mean

K = 5; LAYERS = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']; BAD = ['nan', 'None', 'remove', 'unassigned', 'WM', 'white_matter']


def axes_dir():
    return ensure(os.path.join(cfg().work, 'axes'))


def smoothed_expression(A, sec, xy, lib, SEC):
    """log1p CP10k smoothed over a disk of r 1.5 grid steps, centred per section (used to name axes by genes)."""
    L = sp.diags(1e4 / lib.astype(np.float32)) @ A.X; L = L.tocsr().astype(np.float32); L.data = np.log1p(L.data)
    G = {s: v for s, v in grid_positions(sec, xy).items()}
    S = np.zeros(L.shape, np.float32)
    for s in SEC:
        disk_mean(lambda idx: L[idx].toarray(), L.shape[1], {s: G[s]}, 1.5, chunk=250, out=S, min_count=1)
        idx = G[s][0]; S[idx] -= S[idx].mean(0)
    return S


def run(items, ds=None, tag=''):
    c = cfg(); ds = ds or c.series['name']; OUT = ds + tag; rng = np.random.default_rng(0)
    A = ad.read_h5ad(c.inputs(ds)); o = A.obs; xy = np.asarray(A.obsm['spatial'], float); genes = np.array(A.var_names)
    sec = o.slice.astype(str).values; lay = o.layer.astype(str).values; reg = o.region.astype(str).values; col = o.col.values.astype(int)
    dep = o.depth.values.astype(float); lib = o.library_size.values.astype(float); llib = np.log(lib); snum = np.array([int(s[1:]) for s in sec])
    SEC = sorted(set(sec), key=lambda t: int(t[1:])); NSEC = len(SEC)
    okl = np.isin(lay, LAYERS); okr = ~np.isin(reg, BAD)
    rcount = pd.Series(reg[okl & okr]).value_counts(); GOODREG = list(rcount[rcount >= 30].index); NREG = len(GOODREG)
    C = pd.read_csv(c.columns(ds)); SA = {(s, cc): v for s, cc, v in zip(C.slice.astype(str), C.col, C.Ridge_SA_index)}
    AREA = {(s, cc): a for s, cc, a in zip(C.slice.astype(str), C.col, C.area_level3.astype(str))}
    sa_reg = C.groupby('area_level3').Ridge_SA_index.mean()
    S = smoothed_expression(A, sec, xy, lib, SEC); print('expression smoothed', flush=True)

    def eta(v, g):
        s = pd.Series(v); return float(((s.groupby(g).transform('mean') - s.mean()) ** 2).sum() / ((s - s.mean()) ** 2).sum())

    def eta_reg(v, idx):
        q = idx[np.isin(reg[idx], GOODREG)]
        if NREG < 3 or len(q) < 50: return np.nan
        w = v[np.isin(reg[idx], GOODREG)]; w = w - pd.Series(w).groupby(lay[q]).transform('mean').values; return eta(w, reg[q])

    def secmean_r(v, ss):
        if NSEC < 5: return np.nan
        m = pd.Series(v).groupby(ss).mean(); return abs(np.corrcoef(m.values, [int(x[1:]) for x in m.index])[0, 1])

    def rc_or_offsets(st):
        return 'rostro-caudal gradient' if (NSEC >= 5 and (st.get('trend_R2', 0) >= 0.3 or st.get('secmean_r', 0) >= 0.7)) else 'section offsets'

    def trend(v, num):
        if NSEC < 5: return np.nan
        b = np.polyfit(num, v, 1); return 1 - ((v - np.polyval(b, num)) ** 2).sum() / ((v - v.mean()) ** 2).sum()

    def sa_rho(v, idx):
        g = pd.DataFrame({'s': sec[idx], 'c': col[idx], 'v': v}); g = g[g.c >= 0].groupby(['s', 'c']).v.mean()
        x = np.array([SA.get(kk, np.nan) for kk in g.index]); f = np.isfinite(x)
        return spearmanr(g.values[f], x[f])[0] if f.sum() > 10 else np.nan

    def reg_ends(v, idx):
        q = np.isin(reg[idx], GOODREG)
        if NREG < 3: return '', ''
        w = v[q] - pd.Series(v[q]).groupby(lay[idx][q]).transform('mean').values; m = pd.Series(w).groupby(reg[idx][q]).mean().sort_values()
        return ', '.join(m.index[:3]), ', '.join(m.index[::-1][:3])

    def gene_ends(v, idx=None, Sm=None):
        """Genes most negatively / positively correlated with v over spots idx (subsample 60k) or over the rows of Sm."""
        if Sm is None:
            sub = idx if len(idx) <= 60000 else rng.choice(idx, 60000, replace=False); vv = v[np.searchsorted(idx, sub)] if len(idx) > 60000 else v; Sm = S[sub]
        else: vv = v
        Sz = (Sm - Sm.mean(0)) / (Sm.std(0) + 1e-9); pz = (vv - vv.mean()) / (vv.std() + 1e-12); cg = Sz.T @ pz / len(pz)
        og = np.argsort(cg); lo = [g for g in genes[og] if not g.startswith('LOC')][:12]; hi = [g for g in genes[og[::-1]] if not g.startswith('LOC')][:12]
        return lo, hi

    def prog(lo, hi):
        tl, th = genes_tag(', '.join(lo)), genes_tag(', '.join(hi))
        if tl in KNOWN or th in KNOWN: return f'programme: {tl} ↔ {th}'
        return f'programme: {", ".join(lo[:2])} ↔ {", ".join(hi[:2])}'

    def read_spot(st, kind):
        if st['section_eta2'] >= 0.3: return rc_or_offsets(st)
        if kind == 'spot' and st['layer_eta2'] >= 0.5: return f"depth {st['layer_lo']} ↔ {st['layer_hi']}"
        if kind in ('L2', 'L3') and st['depth_r'] >= 0.35: return f'depth inside {kind}'
        if abs(st['SA_rho']) >= 0.6: return 'SA axis'
        if st['region_eta2'] >= 0.3: return f"region axis {st['reg_lo'].split(', ')[0]} ↔ {st['reg_hi'].split(', ')[0]}"
        if abs(st['UMI_rho']) >= 0.5: return 'sequencing depth (UMI)'
        return prog(st['_glo'], st['_ghi'])

    def spot_stats(v, idx, kind):
        st = dict(section_eta2=eta(v, sec[idx]), trend_R2=trend(v, snum[idx]), secmean_r=secmean_r(v, sec[idx]), SA_rho=sa_rho(v, idx), region_eta2=eta_reg(v, idx),
                  UMI_rho=spearmanr(v, llib[idx])[0] if len(v) < 200000 else spearmanr(v[::3], llib[idx][::3])[0])
        f = np.isfinite(dep[idx]); st['depth_r'] = abs(np.corrcoef(v[f], dep[idx][f])[0, 1]) if f.sum() > 10 else np.nan
        if kind == 'spot':
            st['layer_eta2'] = eta(v, lay[idx]); lm = pd.Series(v).groupby(lay[idx]).mean().sort_values(); st['layer_lo'], st['layer_hi'] = lm.index[0], lm.index[-1]
        st['reg_lo'], st['reg_hi'] = reg_ends(v, idx); st['_glo'], st['_ghi'] = gene_ends(v, idx)
        st['reading'] = read_spot(st, kind); return st

    def pca(X, n=K):
        X = X - X.mean(0); U, s, Vt = np.linalg.svd(X, full_matrices=False); return (U[:, :n] * s[:n]), Vt[:n], s ** 2 / (s ** 2).sum(), X

    rows, cross = [], []
    for mod, emb in items:
        Z0 = np.load(c.emb(ds, emb)).astype(np.float64); keepf = Z0.std(0) > 0; Z = Z0[:, keepf]; F = Z.shape[1]
        out = {'keepf': keepf}

        def add(an, comp, var, st, **kw):
            rows.append(dict(model=mod, analysis=an, comp=comp, var_share=var, **{k: v for k, v in st.items() if not k.startswith('_')},
                             genes_low=', '.join(st.get('_glo', [])[:6]), genes_high=', '.join(st.get('_ghi', [])[:6]), **kw))
        # ---- spot PCA
        idx = np.flatnonzero(okl); P, V, ve, _ = pca(Z[idx]); out['spot_V'] = V; sc = np.full((len(sec), K), np.nan); sc[idx] = P; out['spot_scores'] = sc.astype(np.float32)
        for j in range(K): add('spot', j + 1, ve[j], spot_stats(P[:, j], idx, 'spot'))
        # ---- layer PCAs
        LV, LX, LI = {}, {}, {}
        for Ly in ['L2', 'L3']:
            idx = np.flatnonzero(lay == Ly); P, V, ve, X = pca(Z[idx]); LV[Ly], LX[Ly], LI[Ly] = V, X, idx; out[f'{Ly}_V'] = V
            sc = np.full((len(sec), K), np.nan); sc[idx] = P; out[f'{Ly}_scores'] = sc.astype(np.float32)
            for j in range(K): add(Ly, j + 1, ve[j], spot_stats(P[:, j], idx, Ly))
        # ---- column PCA
        m = (col >= 0) & np.isin(lay, ['L2', 'L3', 'L4', 'L5', 'L6'])
        g = pd.DataFrame({'s': sec[m], 'c': col[m], 'l': lay[m]}).groupby(['s', 'c', 'l']).indices; im = np.flatnonzero(m)
        acc, accS, accL = {}, {}, {}
        for (s, cc, l), ix in g.items():
            if len(ix) >= 3:
                gi = im[ix]; acc.setdefault((s, cc), []).append(Z[gi].mean(0)); accS.setdefault((s, cc), []).append(S[gi].mean(0)); accL.setdefault((s, cc), []).append(llib[gi].mean())
        keys = [k for k, v in acc.items() if len(v) >= 3]
        Pc = np.array([np.mean(acc[k], 0) for k in keys]); Sc = np.array([np.mean(accS[k], 0) for k in keys]); Lc = np.array([np.mean(accL[k]) for k in keys])
        cs = np.array([k[0] for k in keys]); ca = np.array([AREA.get(k, 'nan') for k in keys]); csa = np.array([SA.get(k, np.nan) for k in keys]); cn = np.array([int(k[0][1:]) for k in keys])
        P, V, ve, _ = pca(Pc); out['col_V'] = V; out['col_keys'] = np.array([f'{s}|{cc}' for s, cc in keys]); out['col_scores'] = P.astype(np.float32)
        nar = pd.Series(ca).value_counts(); gar = list(nar[nar >= 3].index)
        for j in range(K):
            v = P[:, j]; f = np.isfinite(csa); qa = np.isin(ca, gar)
            st = dict(section_eta2=eta(v, cs), trend_R2=trend(v, cn), secmean_r=secmean_r(v, cs), SA_rho=spearmanr(v[f], csa[f])[0], UMI_rho=spearmanr(v, Lc)[0])
            if len(gar) >= 3:
                w = v[qa] - pd.Series(v[qa]).groupby(cs[qa]).transform('mean').values; st['region_eta2'] = eta(w, ca[qa])
                am = pd.Series(v[qa]).groupby(ca[qa]).mean().sort_values(); st['reg_lo'], st['reg_hi'] = ', '.join(am.index[:3]), ', '.join(am.index[::-1][:3])
            else: st['region_eta2'], st['reg_lo'], st['reg_hi'] = np.nan, '', ''
            st['_glo'], st['_ghi'] = gene_ends(v, Sm=Sc)
            if st['section_eta2'] >= 0.3: rd = rc_or_offsets(st)
            elif abs(st['SA_rho']) >= 0.6: rd = 'SA axis'
            elif st['region_eta2'] >= 0.5: rd = f"region axis {st['reg_lo'].split(', ')[0]} ↔ {st['reg_hi'].split(', ')[0]}"
            elif abs(st['UMI_rho']) >= 0.5: rd = 'sequencing depth (UMI)'
            else: rd = prog(st['_glo'], st['_ghi'])
            st['reading'] = rd; add('column', j + 1, ve[j], st, n_units=len(keys))
        # ---- pseudobulk areal-axis model
        sd = Z.std(0); Zs = Z / sd; pbd = None
        if NREG >= 6:
            def pb_axes(keep_sec):
                mm = okl & okr & np.isin(sec, keep_sec) & np.isin(reg, GOODREG); df = pd.DataFrame({'t': sec[mm], 'r': reg[mm], 'l': lay[mm]})
                gg = df.groupby(['t', 'r', 'l']).indices; ii = np.flatnonzero(mm); kk = [k for k, v in gg.items() if len(v) >= 5]
                PB = np.array([Zs[ii[gg[k]]].mean(0) for k in kk]); SB = np.array([S[ii[gg[k]]].mean(0) for k in kk]); W = np.array([len(gg[k]) for k in kk], float)
                kt = np.array([k[0] for k in kk]); kr = np.array([k[1] for k in kk]); kl = np.array([k[2] for k in kk])
                for l in LAYERS:                                   # f_l: remove the layer profile
                    q = kl == l
                    if q.any(): PB[q] -= np.average(PB[q], axis=0, weights=W[q]); SB[q] -= np.average(SB[q], axis=0, weights=W[q])
                cc_ = {s: np.log(np.median(lib[sec == s])) for s in SEC}; cmn = np.mean(list(cc_.values())); ct = np.array([cc_[t] - cmn for t in kt])
                th = (W * ct) @ PB / (W * ct * ct).sum(); PB -= ct[:, None] * th[None, :]     # theta c_t: remove sequencing depth
                regs = sorted(set(kr)); M = np.zeros((len(regs), len(LAYERS) * F)); hv = np.zeros((len(regs), len(LAYERS)), bool); SR = np.zeros((len(regs), S.shape[1]))
                for i, r in enumerate(regs):                       # region x (layer, dim) matrix of residual means
                    for j, l in enumerate(LAYERS):
                        q = (kr == r) & (kl == l)
                        if q.any(): M[i, j * F:(j + 1) * F] = np.average(PB[q], axis=0, weights=W[q]); hv[i, j] = True
                    q = kr == r; SR[i] = np.average(SB[q], axis=0, weights=W[q])
                keep = hv.sum(1) >= 3; regs = [r for r, k in zip(regs, keep) if k]; M = M[keep]; M -= M.mean(0); SR = SR[keep]
                U, s_, Vt = np.linalg.svd(M, full_matrices=False); Kp = min(K, len(regs) - 1)   # mu = U sqrt(n), g = Vt
                return pd.DataFrame(U[:, :Kp] * np.sqrt(len(regs)), index=regs), Vt[:Kp], s_ ** 2 / (s_ ** 2).sum(), SR, hv[keep]
            mu, Vt, ve, SR, hv = pb_axes(SEC); mu = (mu - mu.mean()) / mu.std()
            odd = [s for s in SEC if int(s[1:]) % 2]; even = [s for s in SEC if not int(s[1:]) % 2]
            mo, _, _, _, _ = pb_axes(odd); me, _, _, _, _ = pb_axes(even)
            rcn = pd.Series(snum[okl & okr]).groupby(reg[okl & okr]).mean()
            G = Vt.reshape(len(Vt), len(LAYERS), F) / sd[None, None, :]       # loading per layer in the raw latent space
            out['pb_regions'] = np.array(mu.index); out['pb_mu'] = mu.values; out['pb_G'] = G; out['pb_has_layer'] = hv
            for j in range(mu.shape[1]):
                f = mu[j]; ci = mo.index.intersection(f.index); ce = me.index.intersection(f.index); cc_ = mo.index.intersection(me.index)
                bo = max(mo.columns, key=lambda c_: abs(np.corrcoef(mo.loc[ci, c_], f[ci])[0, 1])); be = max(me.columns, key=lambda c_: abs(np.corrcoef(me.loc[ce, c_], f[ce])[0, 1]))
                stab = abs(np.corrcoef(mo.loc[cc_, bo], me.loc[cc_, be])[0, 1]) if len(cc_) >= 4 else np.nan
                sar = sa_reg.reindex(mu.index); q = sar.notna(); sr = np.corrcoef(f[q], sar[q])[0, 1] if q.sum() >= 4 else np.nan
                arch = np.nan
                if j > 0:
                    b = np.polyfit(mu[0], f, 2); arch = 1 - ((f - np.polyval(b, mu[0])) ** 2).sum() / ((f - f.mean()) ** 2).sum()
                rc = abs(np.corrcoef(f, rcn.reindex(mu.index))[0, 1]) if NSEC >= 4 else np.nan
                srt = f.sort_values(); lo, hi = ', '.join(srt.index[:3]), ', '.join(srt.index[::-1][:3])
                glo, ghi = gene_ends(f.values, Sm=SR)
                if abs(sr) >= 0.6: rd = 'SA axis'
                elif j > 0 and arch >= 0.5: rd = 'arch of axis 1'
                elif NSEC >= 4 and rc >= 0.6: rd = 'rostro-caudal'
                elif np.isfinite(stab) and stab < 0.5: rd = f'not reproducible ({srt.index[0]} ↔ {srt.index[-1]})'
                else: rd = f'areal axis {srt.index[0]} ↔ {srt.index[-1]}'
                add('pseudobulk', j + 1, ve[j], dict(SA_r=sr, splithalf=stab, arch_R2=arch, rostrocaudal_r=rc, reg_lo=lo, reg_hi=hi, reading=rd, _glo=glo, _ghi=ghi), n_units=len(mu))
            gl = []
            for j in range(len(Vt)):
                w = G[j][hv.any(0)]; d = w.sum(0); gl.append(d / np.linalg.norm(d))
            pbd = np.array(gl); out['pb_dir'] = pbd
            g23 = []
            for j in range(len(Vt)):
                d = G[j][1] / np.linalg.norm(G[j][1]) + G[j][2] / np.linalg.norm(G[j][2]); g23.append(d / np.linalg.norm(d))
            out['pb_dir_L23'] = np.array(g23)
        # ---- L2 / L3 shared and layer-specific directions
        U2, s2, W2t = np.linalg.svd(LV['L2'] @ LV['L3'].T); null = []
        for _ in range(300):
            a = np.linalg.qr(rng.normal(size=(F, K)))[0].T; b = np.linalg.qr(rng.normal(size=(F, K)))[0].T; null.append(np.linalg.svd(a @ b.T, compute_uv=False)[0])
        cos_null = np.percentile(null, 99)
        CM = {}
        for Ly in ['L2', 'L3']:
            q = col[LI[Ly]] >= 0; CM[Ly] = pd.DataFrame(LX[Ly][q]).groupby([sec[LI[Ly]][q], col[LI[Ly]][q]]).mean()
        com = CM['L2'].index.intersection(CM['L3'].index); A2, A3 = [(CM[Ly].loc[com] - CM[Ly].loc[com].groupby(level=0).transform('mean')).values for Ly in ['L2', 'L3']]
        rr = []
        for _ in range(500):
            d = rng.normal(size=F); d /= np.linalg.norm(d); rr.append(np.corrcoef(A2 @ d, A3 @ d)[0, 1])
        colr_null = np.percentile(rr, 95)
        D = []
        for k in range(K):
            a = U2[:, k] @ LV['L2']; b = W2t[k] @ LV['L3']; d = a + b; d /= np.linalg.norm(d); D.append(d)
            st = {}
            for Ly in ['L2', 'L3']:
                p = LX[Ly] @ d; st[f'var_{Ly}'] = p.var() / LX[Ly].var(0).sum(); st[f'SA_rho_{Ly}'] = sa_rho(p, LI[Ly]); st[f'region_eta2_{Ly}'] = eta_reg(p, LI[Ly])
                f = np.isfinite(dep[LI[Ly]]); st[f'depth_r_{Ly}'] = abs(np.corrcoef(p[f], dep[LI[Ly]][f])[0, 1])
                st[f'section_eta2_{Ly}'] = eta(p, sec[LI[Ly]]); st[f'secmean_r_{Ly}'] = secmean_r(p, sec[LI[Ly]]); st[f'trend_R2_{Ly}'] = trend(p, snum[LI[Ly]])
            cr = np.corrcoef(A2 @ d, A3 @ d)[0, 1]; ii = np.r_[LI['L2'], LI['L3']]; pp = np.r_[LX['L2'] @ d, LX['L3'] @ d]; o_ = np.argsort(ii)
            lo, hi = reg_ends(pp[o_], ii[o_]); glo, ghi = gene_ends(pp[o_], ii[o_])
            best = {}
            for nm, M_ in [('spot', out['spot_V']), ('column', out['col_V'])] + ([('pseudobulk', out['pb_dir_L23'])] if pbd is not None else []):
                cc2 = np.abs(M_ @ d); best[nm] = (int(np.argmax(cc2)) + 1, cc2.max())
            sa_m = np.nanmean([abs(st['SA_rho_L2']), abs(st['SA_rho_L3'])]); reg_both = NREG >= 3 and min(st['region_eta2_L2'], st['region_eta2_L3']) >= 0.3; areal = cr > colr_null or reg_both
            if s2[k] < cos_null: rd = 'not shared'
            elif sa_m >= 0.6: rd = 'SA axis'
            elif (st['section_eta2_L2'] + st['section_eta2_L3']) / 2 >= 0.3:
                rd = rc_or_offsets(dict(trend_R2=(st['trend_R2_L2'] + st['trend_R2_L3']) / 2, secmean_r=(st['secmean_r_L2'] + st['secmean_r_L3']) / 2))
            elif min(st['depth_r_L2'], st['depth_r_L3']) >= 0.35: rd = 'depth inside L2 and L3'
            elif areal and NREG >= 3: rd = f'areal axis {lo.split(", ")[0]} ↔ {hi.split(", ")[0]}'
            elif areal: rd = 'areal pattern (one region; column-level)'
            else: rd = prog(glo, ghi) + ' (not areal)'
            only = s2[k] >= cos_null and areal and max(v[1] for v in best.values()) < 0.5
            add('shared', k + 1, (st['var_L2'] + st['var_L3']) / 2, dict(principal_cos=s2[k], cos_null99=cos_null, column_r=cr, column_r_null95=colr_null, reg_lo=lo, reg_hi=hi,
                reading=rd, L23_only=only, _glo=glo, _ghi=ghi, **st, **{f'best_{n}': f'{v[0]}:{v[1]:.2f}' for n, v in best.items()}))
        out['shared_D'] = np.array(D)
        for Ly, Ub in [('L2', U2), ('L3', W2t.T)]:
            for k in range(K):
                if s2[k] >= 0.6: continue
                d = Ub[:, k] @ LV[Ly]; p = LX[Ly] @ d; st = spot_stats(p, LI[Ly], Ly)
                add(f'{Ly}_only', k + 1, p.var() / LX[Ly].var(0).sum(), st, principal_cos=s2[k])
        # ---- cross-method |cos|
        blocks = [('spot', out['spot_V']), ('column', out['col_V'])] + ([('pseudobulk', pbd)] if pbd is not None else []) + [('L2∩L3', out['shared_D'])]
        names = [f'{n} {i + 1}' for n, M_ in blocks for i in range(len(M_))]; ALL = np.vstack([M_ for _, M_ in blocks]); CC = np.abs(ALL @ ALL.T)
        cross.append(pd.DataFrame(CC, index=names, columns=names).assign(model=mod))
        rn = []
        for _ in range(3000):
            v = rng.normal(size=F); v /= np.linalg.norm(v); Q = np.linalg.qr(rng.normal(size=(F, K)))[0]; rn.append(np.abs(Q.T @ v).max())
        out['cos_null95'] = np.percentile(rn, 95)
        np.savez(os.path.join(axes_dir(), f'{OUT}_{mod.replace(" ", "_")}.npz'), **out)
        print(mod, 'done', flush=True)
    ensure(c.tables())
    R = pd.DataFrame(rows); R.to_csv(c.tables(f'axes_{OUT}_comp.csv'), index=False)
    pd.concat(cross).to_csv(c.tables(f'axes_{OUT}_cross.csv'))
    return R


if __name__ == '__main__':
    P = argparse.ArgumentParser(); P.add_argument('items'); P.add_argument('--ds', default=None); P.add_argument('--tag', default='')
    a = P.parse_args(); run([tuple(x.split(':')) for x in a.items.split(',')], a.ds, a.tag)
