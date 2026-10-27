"""Step 05: (1) copy the final latents of the trained models into work/embeddings/<series>/<model>__seed0.npy (rows checked
against inputs/<series>.h5ad); (2) encode the young brains (P0, P32, 3M, 7M) with every adult FGW 2D model - the adult encoder
applied to new sections, no training (ottr.inspire.train --encode_only --init_ckpt) -> work/embeddings/dev<age>/adultenc_<model>;
(3) write the reconstructions of the FGW 2D models for the R2 (--recon_out) -> work/recon/<model>_*.
GPU for (2) and (3), a few minutes per job.  usage: python scripts/05_collect_encode.py [--skip_encode] [--skip_recon]"""
import os, sys, json, argparse, subprocess
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import anndata as ad
from ottr.config import cfg, ensure
from ottr.registry import MODELS, FGW2D, dev_emb
from ottr.inspire.train import PRESETS

c = cfg(); ds = c.series['name']; py = c['python']['train']
P = argparse.ArgumentParser(); P.add_argument('--skip_encode', action='store_true'); P.add_argument('--skip_recon', action='store_true'); a = P.parse_args()
ref = ad.read_h5ad(c.inputs(ds), backed='r').obs_names.values
genes = os.path.join(c.work, 'inputs', f'{ds}_genes.txt')
open(genes, 'w').write('\n'.join(ad.read_h5ad(c.inputs(ds), backed='r').var_names) + '\n')
for key, (label, emb, trained) in MODELS.items():
    if not trained: continue
    f = os.path.join(c.models(key), 'latent.npz')
    if not os.path.exists(f): print('not trained yet:', key); continue
    L = np.load(f); assert (L['obs_names'] == ref).all(), key
    np.save(c.emb(ds, emb), L['Z'].astype(np.float32)); json.dump(dict(method=key, kind='inspire', dim=int(L['Z'].shape[1])), open(c.emb(ds, emb)[:-4] + '.json', 'w'))
    print('collected', key, L['Z'].shape)


def opts(key):
    """Architecture options of a preset as command-line flags (no FGW loss / spec needed to encode)."""
    p = PRESETS[key]; out = []
    for k in ('decode_axf', 'encode_slice', 'direct_counts'):
        if p.get(k): out += [f'--{k}', 'true']
    if p.get('count_dist'): out += ['--count_dist', p['count_dist']]
    return out + ['--coef_fgw', '0']


def gpu_run(cmd, log):
    print(' '.join(cmd[:4]), '...', flush=True)
    r = subprocess.run(cmd, stdout=open(log, 'w'), stderr=subprocess.STDOUT)
    if r.returncode: sys.exit(f'failed, see {log}')


for key, _, _, _ in FGW2D:
    ck = os.path.join(c.models(key), 'checkpoints', 'ckpt_step20000.pt')
    if not os.path.exists(ck): print('no final checkpoint yet:', key); continue
    if not a.skip_encode:
        for dds in c['development']:
            out = os.path.join(c.models(key), f'encode_{dds}')
            if not os.path.exists(os.path.join(out, 'latent.npz')):
                gpu_run([py, '-m', 'ottr.inspire.train', '--slice_dir', c.run_input(dds), '--force_genes_file', genes, '--init_ckpt', ck, '--encode_only',
                         '--out', out, '--seed', '0'] + opts(key), os.path.join(ensure('logs'), f'encode_{key}_{dds}.log'))
            L = np.load(os.path.join(out, 'latent.npz')); assert (L['obs_names'] == ad.read_h5ad(c.inputs(dds), backed='r').obs_names.values).all()
            ensure(c.emb_dir(dds)); np.save(c.emb(dds, dev_emb(key)), L['Z'].astype(np.float32)); print('encoded', key, dds, L['Z'].shape)
    if not a.skip_recon:
        pre = os.path.join(ensure(os.path.join(c.work, 'recon')), key)
        if not os.path.exists(pre + '_features.npz'):
            gpu_run([py, '-m', 'ottr.inspire.train', '--slice_dir', c.run_input(ds), '--svg_genes_file', c.path('svg_genes'), '--init_ckpt', ck, '--recon_out', pre,
                     '--out', os.path.join(c.models(key), 'recon'), '--seed', '0'] + opts(key), os.path.join(ensure('logs'), f'recon_{key}.log'))
