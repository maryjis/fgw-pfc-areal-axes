"""Task A of the validation plan: the results computed before, brought to the common row format (common.ROW_COLUMNS).
python -m ottr.validation.collect
atlas   E1 (validation/e1_regions.csv: spatial p; e1_regions_perm.csv: permutation p), E2 (e2_match*.csv)
species macaque / human frontal regions vs mu (significance/spatial_null.csv 3D, robustness.csv CI / LOO, gene_null.csv model pool);
        plain permutation p computed here (rows = their regions, our mu carried as in species.frontal)
snrna   snrna_types/significance.csv (exact permutation, Moran, variogram, CI) + gene_null.csv (genome pool, two-sided)
-> results/tables/validation/rows_{atlas,species,snrna}.csv"""
import numpy as np
import pandas as pd
from ..config import cfg, ROOT
from . import common as C

MODALITY = lambda f: 'MRI' if (f in ('Myelinmap', 'thickness_L2-L6', 'L4_fraction') or 'Layer' in f) else 'connectivity' if f.startswith('SG') else \
    'fMRI' if (f.startswith('FG') or 'Network' in f or 'Task' in f) else 'hierarchy'


def atlas():
    c = cfg(); v = C.out_dir(); S = pd.read_csv(f'{v}/e1_regions.csv'); P = pd.read_csv(f'{v}/e1_regions_perm.csv')[['ours', 'feature', 'p_perm', 'family']]
    D = S.merge(P, on=['ours', 'feature'], suffixes=('_s', '')); rows = []
    for _, r in D.iterrows():
        rows.append(dict(domain='atlas maps', dataset=MODALITY(r.feature), species='marmoset', axis=r.ours.split()[-1].replace('axis', '') if 'axis' in r.ours else 'PrAl', ours=r.ours, feature=r.feature, n=r.n,
                         family=r.family, spearman=r.spearman, pearson=r.pearson, ci_low=r.ci_low, ci_high=r.ci_high, p_perm=r.p_perm, p_msr=r.p_msr, p_vario=r.p_variogram,
                         loo_min=r.loo_min, loo_max=r.loo_max, replication=('same' if np.sign(r.rho_outside) == np.sign(r.spearman) else 'opposite') if np.isfinite(r.rho_outside) else np.nan,
                         note=f'outside our series: rho {r.rho_outside:+.2f} (n {int(r.n_outside)}); partial | axis1 {r.partial_axis1:+.2f}' if np.isfinite(r.get('partial_axis1', np.nan)) else f'outside our series: rho {r.rho_outside:+.2f} (n {int(r.n_outside)})'))
    E = pd.read_csv(f'{v}/e2_match.csv'); Ep = pd.read_csv(f'{v}/e2_match_perm.csv')
    for (k, mod), g in E.groupby(['axis', 'modality']):
        m, va = g[g.null == 'msr'].iloc[0], g[g.null == 'variogram'].iloc[0]; pp = Ep[(Ep.axis == k) & (Ep.modality == mod)].iloc[0]
        rows.append(dict(domain='atlas components (E2)', dataset=mod, species='marmoset', axis=str(k), ours=f'mu axis{k}', feature=f'best of {mod} components: {m.best}', n=30, family='primary',
                         spearman=m.rho, p_perm=pp.p_perm, p_msr=m.p_max, p_vario=va.p_max, note='p against the maximum |rho| over the components'))
    return C.save(rows, 'atlas')


def species():
    from ..genescore import species as SP
    c = cfg(); Sn = pd.read_csv(c.tables('gene_axes/significance/spatial_null.csv')); Rb = pd.read_csv(c.tables('gene_axes/significance/robustness.csv'))
    Gn = pd.read_csv(c.tables('gene_axes/significance/gene_null.csv')); FR = pd.read_csv(c.tables('gene_axes/species/compare_frontal.csv'), dtype={'their_region': str})
    rng = np.random.default_rng(5); rows = []
    for sp in ('macaque', 'human'):
        F = FR[FR.species == sp]
        for s in ('PrAl', 'axis1', 'axis2', 'axis3', 'axis4', 'axis5'):
            k = 1 if s == 'PrAl' else int(s[-1]); col = f'mu axis{k}'; mapping = [o.split(',') for o in F.our_regions]
            r = C.test(col, F[f'species_{s}'].values, mapping=mapping, rng=rng)      # perm and spatial on our side, same carrying
            sn = Sn[(Sn.test == 'species frontal') & (Sn.species == sp) & (Sn.ours == f'mu_axis{k}') & (Sn.other == f'{sp} {s}') & (Sn.geometry == '3d')]
            rb = Rb[(Rb.species == sp) & (Rb.ours == f'mu_axis{k}') & (Rb.other == f'{sp} {s}')].iloc[0]; gn = Gn[(Gn.species == sp) & (Gn.score == s) & (Gn.comparison == 'frontal vs our mu') & (Gn.pool == 'model')].iloc[0]
            r.update(domain='species', dataset=('Stereo-seq region pseudobulks (Chen et al. 2023 via Huang et al. 2026)' if sp == 'macaque' else 'microarray HCP-MMP parcels (Huang et al. 2026 archive)'), species=sp,
                     feature=f'{sp} {s} gene score', family='primary' if s in ('axis1', 'axis2', 'axis3', 'axis4', 'axis5') else 'exploratory', p_extra=gn.p_two, p_extra_kind='gene-set null (model pool)',
                     note=f'stored spatial p (msr {sn[sn.method == "msr"].p_spatial.iloc[0]:.3g}, vario {sn[sn.method == "variogram"].p_spatial.iloc[0]:.3g}); exact matches only rho {rb.rho_exact:+.2f} (n {int(rb.n_exact)}); duplicates collapsed rho {rb.rho_collapsed:+.2f}')
            rows.append(r)
    return C.save(rows, 'species')


def snrna():
    c = cfg(); D = pd.read_csv(c.tables('gene_axes/snrna_types/significance.csv')); G = pd.read_csv(c.tables('gene_axes/snrna_types/gene_null.csv')); G = G[G.pool == 'genome']; rows = []
    for _, r in D.iterrows():
        g = G[(G.group == r.group) & (G.score == r.score) & (G.target == r.target)].iloc[0]; k = 'PrAl' if r.score == 'PrAl' else r.score[-1]
        rows.append(dict(domain='snRNA subclasses', dataset=f'adult snRNA-seq, {r.group}', species='marmoset', axis=k, ours=('mu ' if r.target == 'mu' else 'Stereo-seq ') + (r.score if r.score != 'PrAl' else 'axis1 (PrAl genes)'),
                         feature=f'{r.score} score in {r.group}', n=r.n_regions, family=('primary' if r.target == 'mu' and r.score != 'PrAl' else 'exploratory'), spearman=r.rho, ci_low=r.ci_bw_low, ci_high=r.ci_bw_high,
                         p_perm=r.p_perm, p_msr=r.p_msr, p_vario=r.p_vario, p_extra=g.p_two, p_extra_kind='gene-set null (genome pool)',
                         note=f'nuclei-bootstrap CI {r.ci_nuclei_low:+.2f}..{r.ci_nuclei_high:+.2f}; Westfall-Young p {r.p_fwer:.3g}; one sample per region'))
    return C.save(rows, 'snrna')


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    for f in (atlas, species, snrna):
        D = f(); print('##', f.__name__, len(D)); print(D.groupby(['family', 'verdict']).size().unstack(fill_value=0).to_string())
