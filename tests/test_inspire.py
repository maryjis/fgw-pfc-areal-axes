"""The refactored INSPIRE reproduces (1) the original INSPIRE package in the classic mode and (2) the previous project's
benchmark copy in every mode used by the models (FGW joint, no factors Poisson / NB, INSPIRE + GAN baseline): identical weights
after 20 training steps on CPU, identical loss logs."""
import os, sys, json, pickle, subprocess
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = '/workspace/inspire/INSPIRE'                                         # github.com/jiazhao97/INSPIRE (commit 1d7e7a6)
BENCH = '/workspace/Gene_modules/benchmark_alignment_2026_10/envs/src/INSPIRE_bench'
# original names -> ours (the original encoder / decoder are two named layers, ours are ModuleLists)
RENAME = {'encoder_layer1.': 'encoder_layers.0.', 'encoder_layer2.': 'encoder_layers.1.', 'decoder_layer1.': 'decoder_layers.0.', 'decoder_layer2.': 'decoder_layers.1.'}


def sections(n_sec=3, nx=20, ny=12, G=40, seed=0):
    """Per-section raw AnnData of a small grid (counts with layer and region structure)."""
    import anndata as ad, scipy.sparse as sp
    rng = np.random.default_rng(seed); b = rng.normal(-np.log(G), 0.5, G); ads = []
    for s in range(n_sec):
        X, xy = [], []
        for x in range(nx):
            for y in range(ny):
                lg = b + 0.8 * np.sin(y / 3 + np.arange(G)) + 0.5 * np.cos(x / 5 + np.arange(G) * 0.3) + rng.normal(0, 0.1, G) * s
                p = np.exp(lg - lg.max()); p /= p.sum(); X.append(rng.poisson(rng.integers(150, 300) * p)); xy.append((x * 100.0, y * 100.0))
        a = ad.AnnData(sp.csr_matrix(np.array(X, np.float32)), var=pd.DataFrame(index=[f'G{g}' for g in range(G)]))
        a.obs_names = [f's{s}_{i}' for i in range(a.n_obs)]; a.obsm['spatial'] = np.array(xy, np.float32); ads.append(a)
    return ads


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    from ottr.inspire.inputs import build_inputs
    d = tmp_path_factory.mktemp('insp'); ads = sections(); genes = list(ads[0].var_names)
    st_list, _ = build_inputs([f'S{i}' for i in range(len(ads))], ads, genes)
    pickle.dump(st_list, open(d / 'inputs.pkl', 'wb'))
    # a small FGW spec: nodes = 2 x 3 blocks of every section, couplings between neighbouring sections
    rng = np.random.default_rng(1); spec = {'layers': ['all'], 'slices': {}, 'pairs': []}
    for st in st_list:
        names = np.array(st.obs_names); xy = st.obsm['spatial']; blocks = []
        for bx in range(4):
            for by in range(3):
                m = (xy[:, 0] // 500 == bx) & (xy[:, 1] // 400 == by); blocks.append({'all': names[m]})
        spec['slices'][str(st.obs['slice_id'].values[0])] = dict(blocks=blocks, arc=np.arange(len(blocks), dtype=float))
    ids = list(spec['slices'])
    for a, b in zip(ids[:-1], ids[1:]):
        T = rng.random((12, 12)); T /= T.sum(); C = np.abs(np.arange(12)[:, None] - np.arange(12)[None]) / 11.0
        spec['pairs'].append(dict(a=a, b=b, T=T.astype(np.float32), CA=C, CB=C, p=np.full(12, 1 / 12), q=np.full(12, 1 / 12)))
    pickle.dump(spec, open(d / 'spec.pkl', 'wb'))
    return d


def run(impl, d, opt, tag, pkg=None):
    o = dict(n_training_steps=20, batch_size=64, **opt); json.dump(o, open(d / f'{tag}.json', 'w'))
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, '..', 'src'), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'inspire_runner.py'), impl, str(d / 'inputs.pkl'), str(d / f'{tag}.json'), str(d / f'{tag}.npz')] + ([pkg] if pkg else []),
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    return dict(np.load(d / f'{tag}.npz')), (pd.read_csv(d / f'{tag}_loss.csv') if os.path.exists(d / f'{tag}_loss.csv') else None)


def same(a, b, rename=None):
    for k, v in a.items():
        k2 = k
        for x, y in (rename or {}).items(): k2 = k2.replace(x, y)
        assert k2 in b, k2
        np.testing.assert_array_equal(v, b[k2], err_msg=k)


@pytest.mark.skipif(not os.path.isdir(ORIG), reason='original INSPIRE package not available')
def test_classic_equals_original(inputs):
    w_orig, _ = run('orig', inputs, {}, 'orig', ORIG)
    w_new, _ = run('new', inputs, dict(gan_pairs='chain', beta_norm_slices=4), 'classic')
    same(w_orig, w_new, RENAME)


MODES = {
    'fgw_joint_factors': dict(decode_axf=True, coef_gan=0.0, coef_geom=0.0, coef_beta=0.0, coef_patch_var=10.0, coef_fgw=770.0, fgw_mode='joint',
                              fgw_alpha=0.7, fgw_warmup=5, fgw_every=5, fgw_spec_file='spec.pkl'),
    'fgw_direct_poisson': dict(decode_axf=True, coef_gan=0.0, coef_geom=0.0, coef_beta=0.0, direct_counts=True, coef_fgw=770.0, fgw_mode='joint',
                               fgw_alpha=0.7, fgw_warmup=5, fgw_every=5, fgw_spec_file='spec.pkl'),
    'fgw_direct_nb': dict(decode_axf=True, coef_gan=0.0, coef_geom=0.0, coef_beta=0.0, direct_counts=True, count_dist='nb', coef_fgw=770.0, fgw_mode='joint',
                          fgw_alpha=0.7, fgw_warmup=5, fgw_every=5, fgw_spec_file='spec.pkl'),
    'inspire_gan': dict(decode_axf=True, encode_slice=True, coef_gan=1.0, coef_geom=0.0, coef_beta=0.0, coef_patch_var=10.0),
}


@pytest.mark.skipif(not os.path.isdir(BENCH), reason='benchmark copy not available')
@pytest.mark.parametrize('mode', list(MODES))
def test_equals_benchmark_copy(inputs, mode):
    opt = dict(MODES[mode])
    if 'fgw_spec_file' in opt: opt['fgw_spec_file'] = str(inputs / opt['fgw_spec_file'])
    w_b, l_b = run('bench', inputs, opt, f'bench_{mode}', BENCH)
    w_n, l_n = run('new', inputs, opt, f'new_{mode}')
    same(w_b, w_n)
    cols = [c for c in l_b.columns if c in l_n.columns]
    assert len(cols) >= 8
    np.testing.assert_array_equal(l_b[cols].values, l_n[cols].values)
