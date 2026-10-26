"""Level-1 reproduction: the new code gives the previous project's stored results exactly (allclose 1e-7) for every
deterministic step. Inputs, smoothed PCA and the FGW specs are compared in the main work/ directory; everything downstream of
the embeddings in the reference workspace work_ref/ (scripts/tools/run_reference.sh: the new code on the stored embeddings).
Skipped when the previous project's results or the reference workspace are absent."""
import os, json, pickle
import numpy as np
import pandas as pd
import anndata as ad
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'default.yaml')))
OLD = CFG['paths']['reference']; WORK = os.path.join(ROOT, CFG['paths']['work']); REFW = os.path.join(ROOT, 'work_ref'); RT = os.path.join(REFW, 'results', 'tables')
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(OLD, 'results')), reason='previous project not available')
need_ref = pytest.mark.skipif(not os.path.isdir(RT), reason='run scripts/tools/run_reference.sh first')
DEV = ['devP0', 'devP32', 'dev3M', 'dev7M']


def frames_equal(new, old, cols=None, rtol=1e-7, atol=1e-9):
    cols = cols or [c for c in old.columns if c in new.columns]
    assert len(new) == len(old), (len(new), len(old))
    for c in cols:
        a, b = new[c].reset_index(drop=True), old[c].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(b) and pd.api.types.is_numeric_dtype(a):
            np.testing.assert_allclose(a.astype(float), b.astype(float), rtol=rtol, atol=atol, equal_nan=True, err_msg=c)
        else:
            assert (a.astype(str).values == b.astype(str).values).all(), (c, a[a.astype(str) != b.astype(str)].head(3).tolist(), b[a.astype(str) != b.astype(str)].head(3).tolist())


# ------------------------------------------------------------------ inputs, smoothed PCA, FGW anchors (main work dir)
@pytest.mark.parametrize('ds', ['pfcserL'] + DEV)
def test_inputs(ds):
    p = os.path.join(WORK, 'inputs', f'{ds}.h5ad')
    if not os.path.exists(p): pytest.skip('step 01 not run')
    a, b = ad.read_h5ad(p), ad.read_h5ad(f'{OLD}/inputs/{ds}.h5ad')
    assert (a.obs_names == b.obs_names).all() and (a.var_names == b.var_names).all() and (a.X != b.X).nnz == 0
    np.testing.assert_array_equal(a.obsm['spatial'], b.obsm['spatial']); frames_equal(a.obs, b.obs)
    frames_equal(pd.read_csv(os.path.join(WORK, 'inputs', f'{ds}_columns.csv')), pd.read_csv(f'{OLD}/inputs/{ds}_columns.csv'))


@pytest.mark.parametrize('variant', ['none', 'centred', 'harmony'])
def test_smoothed_pca(variant):
    p = os.path.join(WORK, 'embeddings', 'pfcserL', f'curve_r1.5_{variant}__seed0.npy')
    if not os.path.exists(p): pytest.skip('step 02 not run')
    np.testing.assert_array_equal(np.load(p), np.load(f'{OLD}/embeddings/pfcserL/curve_r1.5_{variant}__seed0.npy'))


@pytest.mark.parametrize('spec', ['fgw_spec_pfcserL_a07c_k2.pkl', 'fgw2d_spec_pfcserL_a07c_k2.pkl'])
def test_fgw_specs(spec):
    p = os.path.join(WORK, 'anchors', spec)
    if not os.path.exists(p): pytest.skip('step 03 not run')
    n, o = pickle.load(open(p, 'rb')), pickle.load(open(f'{OLD}/results/anchors/{spec}', 'rb'))
    assert list(n['slices']) == list(o['slices']) and len(n['pairs']) == len(o['pairs'])
    for sf in o['slices']:
        for x, y in zip(n['slices'][sf]['blocks'], o['slices'][sf]['blocks']):
            assert all((np.asarray(x[l]) == np.asarray(y[l])).all() for l in y)
    for p_, q_ in zip(n['pairs'], o['pairs']):
        assert (p_['a'], p_['b']) == (q_['a'], q_['b'])
        for k in ('T', 'CA', 'CB', 'p', 'q'): np.testing.assert_array_equal(p_[k], q_[k])


# ------------------------------------------------------------------ metrics (reference workspace)
EMB3 = ['curve_r1.5_centred', 'encslice_gan1_s20000', 'fgw2d_joint_a07c_c770_k2_s20000']


@need_ref
@pytest.mark.parametrize('emb', EMB3)
def test_evaluate(emb):
    a = json.load(open(f'{REFW}/cache/pfcserL__{emb}__seed0.json'))['metrics']; b = json.load(open(f'{OLD}/results/cache/pfcserL__{emb}__seed0.json'))['metrics']
    assert set(a) == set(b)
    for k in b: assert np.isclose(a[k], b[k], rtol=1e-9, atol=1e-12), (k, a[k], b[k])


@need_ref
def test_nmi_within():
    n = pd.read_csv(f'{RT}/nmi_within_pfcserL.csv').set_index('model').loc[EMB3]; o = pd.read_csv(f'{OLD}/results/nmi_within_pfcserL.csv').set_index('model').loc[EMB3]
    frames_equal(n, o, [c for c in o.columns if c in n.columns])


@need_ref
def test_series_metrics_and_rank_inputs():
    n = pd.read_csv(f'{RT}/series_metrics.csv'); o = pd.read_csv(f'{OLD}/results/series_metrics_rank12.csv').set_index('model').loc[n.model].reset_index()
    km = [c for c in o.columns if c.startswith('nmi_') and c[-2:] in ('L3', 'L5')]        # k-means (OpenMP): last digits vary
    frames_equal(n, o, [c for c in o.columns if c not in km]); frames_equal(n, o, km, rtol=0, atol=2e-3)


# ------------------------------------------------------------------ axes
@need_ref
@pytest.mark.parametrize('tag,models', [('', ['smoothed PCA', 'INSPIRE', 'FGW 1D', 'FGW 2D']), ('_d2d20k', ['FGW 2D']), ('_d2dnb20k', ['FGW 2D'])])
def test_multi_axes(tag, models):
    frames_equal(pd.read_csv(f'{RT}/axes_pfcserL{tag}_comp.csv'), pd.read_csv(f'{OLD}/results/ma/pfcserL{tag}_comp.csv'))
    for m in models:
        a = np.load(f'{REFW}/axes/pfcserL{tag}_{m.replace(" ", "_")}.npz', allow_pickle=True); b = np.load(f'{OLD}/results/ma/pfcserL{tag}_{m.replace(" ", "_")}.npz', allow_pickle=True)
        assert set(a.files) == set(b.files)
        for k in b.files:
            if b[k].dtype.kind in 'fc': np.testing.assert_allclose(a[k], b[k], rtol=1e-7, atol=1e-9, equal_nan=True, err_msg=k)
            else: np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@need_ref
@pytest.mark.parametrize('tag', ['_k2_20k', '_d2d20k', '_d2dnb20k'])
def test_spot_axes(tag):
    n = pd.read_csv(f'{RT}/spot_axes_pfcserL{tag}.csv'); o = pd.read_csv(f'{OLD}/results/spot_axes_pfcserL{tag}.csv'); o = o[o.model == 'FGW 2D']
    frames_equal(n, o)
    for v in ('layer', 'depth', 'layer+depth'):
        np.testing.assert_allclose(np.load(f'{REFW}/axes/spot_axes_scores{tag}_{v}_FGW_2D.npy'), np.load(f'{OLD}/results/spot_axes_scores{tag}_{v}_FGW_2D.npy'), rtol=1e-6, atol=1e-6)


# ------------------------------------------------------------------ atlas
@need_ref
def test_frontal_pca():
    for f in ('frontal_pca_scores.csv', 'frontal_pca_loadings.csv', 'frontal_pca_annotation.csv'):
        frames_equal(pd.read_csv(f'{RT}/atlas/{f}'), pd.read_csv(f'{OLD}/results/flatmaps_frontal/{f}'), rtol=1e-6, atol=1e-8)


@need_ref
def test_flatmap():
    # assignment: the stored one broke ties by Python's random set order -> compare agreement, not identity
    a, b = pd.read_csv(f'{RT}/atlas/spot2segment.csv'), pd.read_csv(f'{OLD}/results/flatmap/spot2segment.csv'); m = a.merge(b, on='spot')
    assert len(m) >= 0.99 * len(b) and (m.segment_x == m.segment_y).mean() >= 0.85
    for d_ in (a, b): assert abs((d_.segment_region == d_.region).mean() - (b.segment_region == b.region).mean()) < 0.01
    # projection and matches: exact, computed on the stored assignment
    for d in ('flatmaps20000', 'flatmaps_d2d20k', 'flatmaps_d2dnb20k'):
        n = pd.read_csv(f'{RT}/atlas/{d}/segment_values.csv'); o = pd.read_csv(f'{OLD}/results/{d}/segment_values.csv')
        frames_equal(n, o, ['segment'] + [c for c in n.columns if c.startswith('FGW 2D|')])
        n = pd.read_csv(f'{RT}/atlas/{d}/segment_matches.csv'); o = pd.read_csv(f'{OLD}/results/{d}/segment_matches.csv'); o = o[o.model == 'FGW 2D']
        frames_equal(n, o)


@need_ref
def test_axes_vs_atlas():
    frames_equal(pd.read_csv(f'{RT}/atlas/axes_vs_frontal_models.csv'), pd.read_csv(f'{OLD}/results/flatmaps_frontal/axes_vs_frontal_models.csv'))


# ------------------------------------------------------------------ development
@need_ref
@pytest.mark.parametrize('new,old', [('dev', 'dev_axes'), ('dev_nf20k', 'dev_axes_nf20k'), ('dev_nb20k', 'dev_axes_nb20k')])
def test_development(new, old):
    O = f'{OLD}/results/{old}'
    for ds in DEV:
        o = pd.read_csv(f'{O}/{ds}_columns.csv'); o = o[o.latent == 'adult encoder (A)']; frames_equal(pd.read_csv(f'{RT}/{new}/{ds}_columns.csv'), o)
        o = pd.read_csv(f'{O}/{ds}_summary.csv'); o = o[o.latent == 'adult encoder (A)']; frames_equal(pd.read_csv(f'{RT}/{new}/{ds}_summary.csv'), o)
    for f in ('conservation.csv', 'conservation_pairs.csv', 'age_areal_samples_median.csv', 'age_areal_genes_median.csv', 'age_areal_rho_median.csv', 'age_areal_metrics.csv'):
        if os.path.exists(f'{O}/{f}'): frames_equal(pd.read_csv(f'{RT}/{new}/{f}'), pd.read_csv(f'{O}/{f}'))
