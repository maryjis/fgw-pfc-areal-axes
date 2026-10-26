"""Step 11: level-2 reproduction check - the RETRAINED models against the stored results of the previous project.
Training is stochastic (GPU, minibatches) and the old factor models were trained as 7 500 steps + a continuation to 20 000, the
new ones in one 20 000-step run, so the check uses tolerances:
  metrics  : every ranking metric (and the others) |new - old| <= 0.02
  axes     : areal-axis model mu over regions and column-PCA scores: for every old axis 1-5 the best |r| with a new axis 1-5
             (axes may swap order or sign); pass if >= 0.95 for the reproducible axes (old split-half >= 0.5)
  atlas    : rho of every areal axis with fPC1-10, |new - old| of the best-matching fPC's rho
  develop  : conservation of the areal axes per age (share, r of the area means), |new - old|
-> results/tables/reproduction_report.csv (one row per check) and reproduction_summary.csv"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from ottr.config import cfg
from ottr.registry import MODELS, RANK_SET, FGW2D, axes_model
from ottr.areal.multi_axes import axes_dir

c = cfg(); ds = c.series['name']; R = lambda *p: c.ref('results', *p); rows = []
OLD_LABEL = {'smoothed_pca': 'smoothed PCA', 'inspire_gan': 'INSPIRE 20000', 'fgw1d_factors': 'FGW 1D factors 20000', 'fgw2d_factors': 'FGW 2D factors 20000',
             'fgw2d_poisson': 'FGW 2D no factors λ770 20000', 'fgw2d_nb': 'FGW 2D no factors NB 20000'}
OLD_AXES = {'smoothed_pca': ('pfcserL', 'smoothed PCA'), 'inspire_gan': ('pfcserL', 'INSPIRE'), 'fgw1d_factors': ('pfcserL', 'FGW 1D'), 'fgw2d_factors': ('pfcserL', 'FGW 2D'),
            'fgw2d_poisson': ('pfcserL_d2d20k', 'FGW 2D'), 'fgw2d_nb': ('pfcserL_d2dnb20k', 'FGW 2D')}
OLD_DEV = {'fgw2d_factors': 'dev_axes', 'fgw2d_poisson': 'dev_axes_nf20k', 'fgw2d_nb': 'dev_axes_nb20k'}
NEW_RUN = {k: (tag, lab) for tag, items in __import__('ottr.registry', fromlist=['AXES_RUNS']).AXES_RUNS.items() for lab, k in items}
TOL_M, TOL_R = 0.02, 0.95

# ---- metrics
new = pd.read_csv(c.tables('series_metrics.csv')).set_index('model'); old = pd.read_csv(R('series_metrics_rank12.csv')).set_index('model')
for k in RANK_SET:
    lab = MODELS[k][0]
    if lab not in new.index: continue
    for m in [x for x in old.columns if x != 'emb' and x in new.columns]:
        d = float(new.loc[lab, m] - old.loc[OLD_LABEL[k], m])
        rows.append(dict(check='metric', model=k, item=m, old=old.loc[OLD_LABEL[k], m], new=new.loc[lab, m], value=d, tolerance=f'|Δ| ≤ {TOL_M}', passed=abs(d) <= TOL_M))

# ---- axes (areal model and column PCA)
for k in RANK_SET:
    if k not in NEW_RUN: continue
    tag, lab = NEW_RUN[k]; f = os.path.join(axes_dir(), f'{ds}{tag}_{lab.replace(" ", "_")}.npz')
    if not os.path.exists(f): continue
    Zn = np.load(f, allow_pickle=True); otag, olab = OLD_AXES[k]; Zo = np.load(R('ma', f'{otag}_{olab.replace(" ", "_")}.npz'), allow_pickle=True)
    To = pd.read_csv(R('ma', f'{otag}_comp.csv')); To = To[(To.model == olab) & (To.analysis == 'pseudobulk')].set_index('comp')
    mo = pd.DataFrame(Zo['pb_mu'], index=Zo['pb_regions']); mn = pd.DataFrame(Zn['pb_mu'], index=Zn['pb_regions']); ix = mo.index.intersection(mn.index)
    for j in range(min(5, mo.shape[1])):
        rr = [abs(np.corrcoef(mo.loc[ix, j], mn.loc[ix, q])[0, 1]) for q in range(min(5, mn.shape[1]))]; b = int(np.argmax(rr))
        repro = To.loc[j + 1].splithalf >= 0.5
        rows.append(dict(check='areal axis', model=k, item=f'old axis {j + 1}: {To.loc[j + 1].reading}', old=j + 1, new=b + 1, value=rr[b], tolerance=f'|r| ≥ {TOL_R} (reproducible axes)',
                         passed=bool(rr[b] >= TOL_R) if repro else None, note='' if repro else f'old split-half {To.loc[j + 1].splithalf:.2f} < 0.5: not required'))
    co = dict(zip(Zo['col_keys'], Zo['col_scores'])); cn = dict(zip(Zn['col_keys'], Zn['col_scores'])); kk = sorted(set(co) & set(cn))
    A, Bn = np.array([co[x] for x in kk]), np.array([cn[x] for x in kk])
    for j in range(5):
        rr = [abs(np.corrcoef(A[:, j], Bn[:, q])[0, 1]) for q in range(5)]; b = int(np.argmax(rr))
        rows.append(dict(check='column PC', model=k, item=f'old column PC {j + 1}', old=j + 1, new=b + 1, value=rr[b], tolerance=f'|r| ≥ {TOL_R}', passed=bool(rr[b] >= TOL_R) if j < 3 else None,
                         note='' if j < 3 else 'PC 4-5: reported only'))

# ---- atlas
fa = c.tables('atlas/axes_vs_frontal_models.csv')
if os.path.exists(fa):
    An = pd.read_csv(fa); Ao = pd.read_csv(R('flatmaps_frontal', 'axes_vs_frontal_models.csv'))
    for k, _, ml, _ in FGW2D:
        for j in range(1, 6):
            a = Ao[(Ao.model == ml) & (Ao.analysis == 'pseudobulk areal') & (Ao.comp == j)]; n = An[(An.model == ml) & (An.analysis == 'pseudobulk areal')]
            if a.empty or n.empty: continue
            a = a.iloc[0]; fb = int(a.best_fPC); best_new = n[f'rho_fPC{fb}'].abs().max()
            rows.append(dict(check='atlas', model=k, item=f'old areal axis {j} ↔ fPC{fb}', old=abs(a.rho_best), new=best_new, value=best_new - abs(a.rho_best),
                             tolerance='|Δ| ≤ 0.1 (best new axis on the same fPC)', passed=bool(abs(best_new - abs(a.rho_best)) <= 0.1)))

# ---- development
for k, _, _, _ in FGW2D:
    M = axes_model(k); f = M.out('conservation.csv')
    if not os.path.exists(f): continue
    Cn = pd.read_csv(f); Co = pd.read_csv(R(OLD_DEV[k], 'conservation.csv'))
    for an in ('pseudobulk',):
        for age in ('P0', 'P32', '3M', '7M'):
            for j in (1, 2, 3):
                o_ = Co[(Co.analysis == an) & (Co.age == age) & (Co.comp == j)].iloc[0]; n_ = Cn[(Cn.analysis == an) & (Cn.age == age) & (Cn.comp == j)].iloc[0]
                rows.append(dict(check='development', model=k, item=f'{age} areal axis {j} share', old=o_.share, new=n_.share, value=n_.share - o_.share, tolerance='reported', passed=None))

T = pd.DataFrame(rows); T.to_csv(c.tables('reproduction_report.csv'), index=False)
S = T[T.passed.notna()].groupby(['check', 'model']).passed.agg(['sum', 'count']).reset_index(); S['all_passed'] = S['sum'] == S['count']
S.to_csv(c.tables('reproduction_summary.csv'), index=False); pd.set_option('display.width', 200); print(S.to_string(index=False))
print('failed checks:'); print(T[T.passed == False][['check', 'model', 'item', 'old', 'new', 'value']].round(3).to_string(index=False))
