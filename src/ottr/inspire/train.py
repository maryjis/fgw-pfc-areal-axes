"""Train (or apply) one INSPIRE / FGW model.  python -m ottr.inspire.train --preset fgw2d_factors --out results/models/fgw2d_factors

Presets (the models of the project; any option can still be overridden on the command line):
  classic        original INSPIRE: adversarial alignment with a chain of S-1 discriminators, geometry and Dirichlet losses,
                 Poisson counts through 30 spatial factors, beta_loss normalised as in the original code (batch x 4)
  inspire_gan    the project's INSPIRE baseline: as classic but all-pairs discriminators, section one-hot in the encoder,
                 smoothed features also reconstructed, patch-variance anti-collapse instead of the geometry / Dirichlet terms
  fgw1d_factors  FGW alignment of (column x layer) blocks, joint coupling (alpha 0.7), lambda 770, no GAN, 30 factors
  fgw2d_factors  the same with (column x depth-bin) blocks (FGW 2D)                     <- best model of the benchmark
  fgw2d_poisson  FGW 2D, counts decoded directly from the latent (no factors), Poisson
  fgw2d_nb       FGW 2D, no factors, negative binomial counts
FGW presets need --fgw_spec (built by scripts/03_fgw_anchors.py).

Modes besides training:
  --encode_only        encode the sections with --init_ckpt (no training): latent of new data in an existing model's space
  --recon_out PREFIX   with --init_ckpt: write predicted / observed log1p CP10k counts (float16 memmaps) and per-gene
                       feature-reconstruction sums, for the R2 of the reconstructions (ottr.metrics.recon_r2)

Outputs in --out:  latent.npz (Z, beta, obs_names, section ids), loss_log.csv, checkpoints/ckpt_step<N>.pt + latent_step<N>.npz,
                   fgw_T.npz + fgw_history.csv (FGW models), gene_loadings.csv (factor models), args.json"""
import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd

PRESETS = {
    'classic': dict(coef_gan=1.0, coef_geom=0.02, coef_beta=1.0, gan_pairs='chain', beta_norm_slices=4),
    'inspire_gan': dict(coef_gan=1.0, coef_geom=0.0, coef_beta=0.0, decode_axf=True, encode_slice=True, coef_patch_var=10.0),
    'fgw1d_factors': dict(coef_gan=0.0, coef_geom=0.0, coef_beta=0.0, decode_axf=True, coef_patch_var=10.0,
                          coef_fgw=770.0, fgw_mode='joint', fgw_alpha=0.7),
    'fgw2d_factors': dict(coef_gan=0.0, coef_geom=0.0, coef_beta=0.0, decode_axf=True, coef_patch_var=10.0,
                          coef_fgw=770.0, fgw_mode='joint', fgw_alpha=0.7),
    'fgw2d_poisson': dict(coef_gan=0.0, coef_geom=0.0, coef_beta=0.0, decode_axf=True, coef_patch_var=0.0, direct_counts=True,
                          coef_fgw=770.0, fgw_mode='joint', fgw_alpha=0.7),
    'fgw2d_nb': dict(coef_gan=0.0, coef_geom=0.0, coef_beta=0.0, decode_axf=True, coef_patch_var=0.0, direct_counts=True, count_dist='nb',
                     coef_fgw=770.0, fgw_mode='joint', fgw_alpha=0.7),
}
DEFAULTS = dict(n_spatial_factors=30, n_training_steps=20000, batch_size=1024, eval_batch_size=4096, hidden_dims=[512, 32],
                coef_recon=1.0, coef_fe=1.0, coef_gan=1.0, coef_geom=0.02, coef_beta=1.0, coef_patch_var=0.0, decode_axf=False,
                encode_slice=False, direct_counts=False, count_dist='poisson', gan_pairs='all', beta_norm_slices=None,
                coef_fgw=0.0, fgw_mode='static', fgw_tau=0.1, fgw_alpha=0.3, fgw_warmup=2000, fgw_every=500, fgw_spots_per_block=8,
                rad_coef=1.5, neighbor_k=8, seed=0)


def parse(argv=None):
    P = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument('--preset', choices=sorted(PRESETS), default=None)
    P.add_argument('--slice_dir', required=True, help='directory of per-section raw h5ad files (work/runs/input/<ds>)')
    P.add_argument('--glob_pattern', default='*_b100_filtered.h5ad')
    P.add_argument('--svg_genes_file', default=None)
    P.add_argument('--force_genes_file', default=None, help='use exactly these genes (zero columns where missing)')
    P.add_argument('--out', required=True)
    P.add_argument('--fgw_spec', default=None)
    P.add_argument('--checkpoint_steps', type=int, nargs='*', default=[2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
    P.add_argument('--loss_log_interval', type=int, default=10)
    P.add_argument('--resume_ckpt', default=None)
    P.add_argument('--init_ckpt', default=None, help='initialise from another run (shape-matched tensors; section embedding = mean)')
    P.add_argument('--encode_only', action='store_true')
    P.add_argument('--recon_out', default=None)
    for k, v in DEFAULTS.items():   # every model option, default from DEFAULTS, then the preset, then the command line
        if isinstance(v, bool): P.add_argument(f'--{k}', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=None)
        elif isinstance(v, list): P.add_argument(f'--{k}', type=int, nargs='+', default=None)
        elif v is None: P.add_argument(f'--{k}', type=int, default=None)
        else: P.add_argument(f'--{k}', type=type(v), default=None)
    a = P.parse_args(argv)
    opt = dict(DEFAULTS); opt.update(PRESETS.get(a.preset, {}))
    opt.update({k: getattr(a, k) for k in DEFAULTS if getattr(a, k) is not None})
    return a, opt


def init_from(model, path):
    """Copy the shape-matched tensors of another checkpoint; section embeddings of a different number of sections become
    the mean of the old ones (gamma stays 0) - the trained encoder is then applied to new sections."""
    import torch
    ck = torch.load(path, map_location=model.device, weights_only=False)['net']; sd = model.net.state_dict(); n_ok, n_new = 0, []
    for k, v in ck.items():
        if k in sd and sd[k].shape == v.shape: sd[k] = v; n_ok += 1
        elif k in sd and k.endswith('slice_emb.weight'): sd[k] = v.float().mean(0, keepdim=True).expand_as(sd[k]).clone(); n_new.append(k)
        elif k in sd: n_new.append(k)
    model.net.load_state_dict(sd); print(f'init from {path}: {n_ok} tensors copied, re-initialised: {n_new}')


def write_recon(model, st_list, genes, ids, prefix, eval_batch_size, decode_axf, direct_counts):
    """Predicted count means and observed counts as log1p(CP10k) (float16, sections concatenated) + per-gene sums of the
    feature reconstruction (genes = own log expression, axf = smoothed) for R2."""
    import torch, torch.nn.functional as F
    net = model.net; net.eval(); dev = model.device; G = len(genes); n_tot = sum(st.n_obs for st in st_list)
    Pm = np.lib.format.open_memmap(f'{prefix}_counts_pred.npy', mode='w+', dtype=np.float16, shape=(n_tot, G))
    Om = np.lib.format.open_memmap(f'{prefix}_counts_obs.npy', mode='w+', dtype=np.float16, shape=(n_tot, G))
    acc = {k: dict(s1=np.zeros(G), s2=np.zeros(G), sse=np.zeros(G), n=0) for k in ('genes', 'axf')}; off = 0
    with torch.no_grad():
        basis = net.get_basis() if not direct_counts else None
        for i, st in enumerate(st_list):
            nf_all = st.obsm['node_features']; cnt = st.obsm['count']; lib = st.obs['library_size'].values.astype(np.float32)
            for b0 in range(0, st.n_obs, eval_batch_size):
                sl = slice(b0, min(st.n_obs, b0 + eval_batch_size)); nf = torch.from_numpy(nf_all[sl]).float().to(dev)
                lab = torch.full((nf.shape[0],), i, dtype=torch.long, device=dev); Zb = net.encoder(nf, lab); emb = net.slice_emb(lab)
                rec = {'genes': net.decoder(Zb, emb)}
                if decode_axf: rec['axf'] = net.decoder2(Zb, emb)
                tgt = {'genes': nf[:, :G], 'axf': nf[:, G:]}
                for k, r in rec.items():
                    y = tgt[k].double().cpu().numpy(); e = y - r.double().cpu().numpy(); a_ = acc[k]
                    a_['s1'] += y.sum(0); a_['s2'] += (y ** 2).sum(0); a_['sse'] += (e ** 2).sum(0); a_['n'] += len(y)
                if direct_counts: log_lam = F.log_softmax(net.count_logits(Zb, emb), dim=1) + net.gamma[lab]
                else:
                    beta, _ = net.deconvolutioner(Zb, emb); log_lam = torch.log(beta @ basis + 1e-6) + net.gamma[lab]
                Pm[off + sl.start: off + sl.stop] = np.log1p(1e4 * torch.exp(log_lam).cpu().numpy()).astype(np.float16)
                Om[off + sl.start: off + sl.stop] = np.log1p(1e4 * cnt[sl] / lib[sl, None]).astype(np.float16)
            off += st.n_obs; print('reconstructed', ids[i], flush=True)
    Pm.flush(); Om.flush()
    np.savez(f'{prefix}_features.npz', genes=np.array(genes), **{f'{k}_{q}': v for k, d_ in acc.items() for q, v in d_.items()},
             slice_sizes=np.array([st.n_obs for st in st_list]), slice_ids=np.array(ids))


def main(argv=None):
    a, opt = parse(argv)
    from .inputs import load_sections, select_genes, build_inputs
    from .model import Model_LGCN
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(vars(a), options=opt), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    ids, ads, shared = load_sections(a.slice_dir, a.glob_pattern)
    genes = select_genes(ads, shared, a.svg_genes_file, a.force_genes_file)
    st_list, adata_full = build_inputs(ids, ads, genes, opt['rad_coef'], opt['neighbor_k'])
    spec = pickle.load(open(a.fgw_spec, 'rb')) if (a.fgw_spec and opt['coef_fgw'] > 0) else None
    if opt['coef_fgw'] > 0 and spec is None: sys.exit('this preset needs --fgw_spec')
    model = Model_LGCN(adata_st_list=st_list, n_spatial_factors=opt['n_spatial_factors'], n_training_steps=opt['n_training_steps'],
                       batch_size=opt['batch_size'], hidden_dims=opt['hidden_dims'], coef_recon=opt['coef_recon'], coef_geom=opt['coef_geom'],
                       coef_fe=opt['coef_fe'], coef_gan=opt['coef_gan'], coef_beta=opt['coef_beta'], decode_axf=opt['decode_axf'],
                       encode_slice=opt['encode_slice'], coef_patch_var=opt['coef_patch_var'], direct_counts=opt['direct_counts'],
                       count_dist=opt['count_dist'], fgw_spec=spec, coef_fgw=opt['coef_fgw'], fgw_mode=opt['fgw_mode'], fgw_tau=opt['fgw_tau'],
                       fgw_alpha=opt['fgw_alpha'], fgw_warmup=opt['fgw_warmup'], fgw_every=opt['fgw_every'],
                       fgw_spots_per_block=opt['fgw_spots_per_block'], gan_pairs=opt['gan_pairs'], beta_norm_slices=opt['beta_norm_slices'],
                       seed=opt['seed'])
    if a.init_ckpt: init_from(model, a.init_ckpt)
    if a.recon_out:
        write_recon(model, st_list, genes, ids, a.recon_out, opt['eval_batch_size'], opt['decode_axf'], opt['direct_counts']); return
    if not a.encode_only:
        model.step_interval = max(1, opt['n_training_steps'] // 10)
        ck = os.path.join(a.out, 'checkpoints') if a.checkpoint_steps else None
        model.train(st_list, loss_log_path=os.path.join(a.out, 'loss_log.csv'), loss_log_interval=a.loss_log_interval,
                    checkpoint_steps=a.checkpoint_steps, checkpoint_dir=ck, checkpoint_eval_batch_size=opt['eval_batch_size'], resume_ckpt=a.resume_ckpt)
        if model.fgw is not None:
            np.savez_compressed(os.path.join(a.out, 'fgw_T.npz'), **{f"T_{P['a']}|{P['b']}": P['T'] for P in model.fgw['pairs']},
                                **{f"T0_{P['a']}|{P['b']}": P['T0'] for P in model.fgw['pairs']})
            pd.DataFrame(model.fgw['history']).to_csv(os.path.join(a.out, 'fgw_history.csv'), index=False)
    adata_full, basis = model.eval_minibatch(st_list, adata_full, batch_size=opt['eval_batch_size'])
    beta = adata_full.obs[[c for c in adata_full.obs.columns if c.startswith('Proportion of spatial factor')]].values.astype(np.float32)
    np.savez_compressed(os.path.join(a.out, 'latent.npz'), Z=np.asarray(adata_full.obsm['latent'], np.float32), beta=beta,
                        obs_names=np.asarray(adata_full.obs_names, dtype='U'), slice_id=np.asarray(adata_full.obs.slice_id, dtype='U'))
    if not opt['direct_counts']:
        basis.to_csv(os.path.join(a.out, 'gene_loadings.csv'))
    print('Done.')


if __name__ == '__main__':
    main()
