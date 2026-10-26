"""GPU: the new training entry point reproduces the previous project's training script bit for bit - 30 steps of the FGW 2D
joint model with spatial factors on the first six sections of the series (sections T405-T407 carry anchors), FGW recomputation
switched on at steps 10 and 20. Compares the latent of every spot, the loss log, the FGW recomputation history and the gene
loadings. Needs a GPU with ~10 GB, the raw sections and the previous project (skipped otherwise). ~4 min."""
import os, subprocess, sys
import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'default.yaml'))); OLD = CFG['paths']['reference']
SCRIPT = f'{OLD}/scripts/inspire_axf_lambda.py'; BENCH = f'{OLD}/envs/src/INSPIRE_bench'; SPEC = f'{OLD}/results/anchors/fgw2d_spec_pfcserL_a07c_k2.pkl'


def gpu():
    try: import torch; return torch.cuda.is_available()
    except Exception: return False


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not (os.path.exists(SCRIPT) and os.path.exists(SPEC)), reason='previous project not available')
@pytest.mark.skipif(not gpu(), reason='no GPU')
def test_training_matches_previous_script(tmp_path):
    import anndata as ad
    inp = tmp_path / 'in'; inp.mkdir(); (tmp_path / 'old').mkdir()
    for t in ['T402', 'T403', 'T404', 'T405', 'T406', 'T407']:     # the first six sections keep the section index of the spec's spot ids
        os.symlink(f"{CFG['paths']['raw_adult']}/{t}_b100_anndata.h5ad", inp / f'{t}_adult_Sub-CTX_b100_filtered.h5ad')
    common = ['--n_training_steps', '30', '--seed', '0', '--fgw_warmup', '10', '--fgw_every', '10', '--checkpoint_steps', '30']
    py = CFG['python']['train']; env = dict(os.environ, INSPIRE_PATH=BENCH)
    old = [py, SCRIPT, '--slice_dir', str(inp), '--glob_pattern', '*_b100_filtered.h5ad', '--svg_genes_file', os.path.join(ROOT, CFG['paths']['svg_genes']),
           '--rs_output', str(tmp_path / 'old'), '--repo_output', str(tmp_path / 'old/repo'), '--results_inspire', str(tmp_path / 'old/f'), '--decode_axf', '--coef_geom', '0',
           '--coef_beta', '0', '--n_spatial_factors', '30', '--coef_patch_var', '10', '--coef_gan', '0', '--fgw_spec', SPEC, '--fgw_mode', 'joint', '--fgw_alpha', '0.7',
           '--coef_fgw', '770', '--run_name', 't'] + common
    new = [py, '-m', 'ottr.inspire.train', '--preset', 'fgw2d_factors', '--slice_dir', str(inp), '--svg_genes_file', os.path.join(ROOT, CFG['paths']['svg_genes']),
           '--fgw_spec', SPEC, '--out', str(tmp_path / 'new')] + common
    for cmd in (old, new):
        r = subprocess.run(cmd, capture_output=True, text=True, env=dict(env, PYTHONPATH=os.path.join(ROOT, 'src'))); assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    o = ad.read_h5ad(tmp_path / 'old/t_inspire_integrated.h5ad'); n = np.load(tmp_path / 'new/latent.npz')
    assert (o.obs_names.values == n['obs_names']).all()
    np.testing.assert_array_equal(np.asarray(o.obsm['latent']), n['Z'])
    a, b = pd.read_csv(tmp_path / 'old/t_loss_log.csv'), pd.read_csv(tmp_path / 'new/loss_log.csv'); cols = [x for x in a.columns if x in b.columns]
    np.testing.assert_array_equal(a[cols].values, b[cols].values)
    assert pd.read_csv(tmp_path / 'old/t_fgw_history.csv').equals(pd.read_csv(tmp_path / 'new/fgw_history.csv'))
    np.testing.assert_array_equal(pd.read_csv(tmp_path / 'old/t_inspire_gene_loadings.csv', index_col=0).values, pd.read_csv(tmp_path / 'new/gene_loadings.csv', index_col=0).values)
