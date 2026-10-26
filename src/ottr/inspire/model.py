"""INSPIRE (LGCN) model with FGW column-anchor alignment - the training objective of every model of this project.

Original: Zhao et al., INSPIRE, INSPIRE/model.py, class Model_LGCN. What is kept from the original and what was added:

  loss = coef_fe    * features_loss     auto-encoder: || x_lognorm - decoder(Z) ||  (+ || A x - decoder2(Z) || with decode_axf)
       + coef_recon * recon_loss        -log p(counts | rate), rate = library * lambda * exp(gamma_section)
                                          lambda = beta @ basis (spatial factors)   or softmax(count logits) (direct_counts)
                                          p = Poisson (original) or negative binomial with a per-gene dispersion (count_dist='nb')
       + coef_geom  * geom_loss         cosine geometry of the batch in Z should match the cosine geometry of the input
       + coef_beta  * beta_loss         Dirichlet(5) prior on the mean factor usage (keeps all factors in use)
       + coef_patch_var * patch_var_loss  extension: -sum_k log Var_batch(patch-averaged beta_k)  (spot + 8 grid neighbours),
                                          rewards factors that vary over the tissue (anti-collapse of the factors)
       + coef_gan   * gan_loss          adversarial section alignment: discriminators try to tell sections apart in Z
                                          (original: chain of S-1 discriminators; here also all C(S,2) pairs, gan_pairs='all')
       + coef_fgw   * fgw_loss          extension (the core of this project): FGW column-anchor alignment, see below

FGW alignment. A precomputed spec (ottr.fgw) lists for every section its nodes - (column x layer) blocks (FGW 1D) or
(column x depth-bin) blocks (FGW 2D) - and for neighbouring section pairs a fused Gromov-Wasserstein coupling T between the
nodes of the two sections (features: node means of the smoothed-expression PCA; structure: distances along the cortex /
in the 2D column x depth plane). During training the latent mean of every node is estimated from a few sampled spots and a
contrastive loss pulls the node of section A towards the nodes of section B that T assigns to it:
    loss_AB = - sum_j T_ij/T_i. log softmax_j( cos(z_A,i , z_B,j) / tau )   (and B -> A), averaged over pairs and layers.
In 'joint' mode T is recomputed every fgw_every steps from the current latent (balanced FGW, structure weight fgw_alpha),
so the coupling and the latent are refined together.

Classic INSPIRE: gan_pairs='chain', beta_norm_slices=4 and all extensions off reproduce the original training step exactly
(tests/test_inspire_classic.py compares the losses with the original package)."""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from .networks import IntegrationNet_LGCN, DiscriminatorNet

LOSS_COLUMNS = ['step', 'loss_total', 'recon_loss', 'fe_loss', 'geom_loss', 'beta_loss', 'patch_var_loss', 'gan_loss', 'd_loss', 'fgw_loss']


def _val(x):
    return x.item() if hasattr(x, 'item') else float(x)


class Model_LGCN:
    def __init__(self,
                 adata_st_list,          # one AnnData per section (ottr.inspire.inputs.build_inputs): obsm node_features, count, neighbor_idx
                 n_spatial_factors,      # K spatial factors
                 n_training_steps,
                 batch_size,             # spots per section per step
                 hidden_dims=(512, 32),
                 coef_recon=1.0, coef_geom=0.02, coef_fe=1.0, coef_gan=1.0, coef_beta=1.0,
                 use_margin=True,        # discriminator logits clamped to +-5 after 100 steps
                 lr_d=5e-4,
                 different_platforms=False,
                 decode_axf=False,       # extension: reconstruct the smoothed features too
                 encode_slice=False,     # extension: section one-hot into the encoder
                 coef_patch_var=0.0,     # extension: patch-variance anti-collapse of the factors
                 direct_counts=False,    # extension: counts decoded directly from Z (no spatial factors)
                 count_dist='poisson',   # extension: 'poisson' | 'nb'
                 fgw_spec=None,          # extension: FGW anchor spec (dict, ottr.fgw.spec1d / spec2d)
                 coef_fgw=0.0,
                 fgw_mode='static',      # 'static': T fixed; 'joint': T recomputed on the latent
                 fgw_tau=0.1,            # temperature of the cosine softmax
                 fgw_alpha=0.3,          # FGW structure weight of the joint recomputation
                 fgw_warmup=2000,        # joint mode: first recomputation step
                 fgw_every=500,          # joint mode: recomputation interval
                 fgw_spots_per_block=8,  # spots sampled per node per step
                 gan_pairs='all',        # 'all': C(S,2) discriminators (rescaled to S-1); 'chain': original S-1 neighbours
                 beta_norm_slices=None,  # beta_loss divides by batch_size * this (None = number of sections; original code: 4)
                 seed=1234):
        self.n_slices = len(adata_st_list)
        self.n_hidden_d = 512; self.slice_emb_dim = 4; self.lr = 5e-4; self.weight_decay = 1e-4; self.weight_decay_d = 1e-4
        self.step_interval = 500
        self.coef_fe, self.coef_gan, self.coef_beta = coef_fe, coef_gan, coef_beta
        self.coef_recon, self.coef_geom, self.coef_patch_var = coef_recon, coef_geom, coef_patch_var
        self.n_spatial_factors = n_spatial_factors; self.n_training_steps = n_training_steps; self.batch_size = batch_size
        self.hidden_dims = list(hidden_dims)
        self.n_features_lgcn = adata_st_list[0].obsm['node_features'].shape[1]
        self.n_hvgs = adata_st_list[0].obsm['count'].shape[1]
        self.use_margin = use_margin; self.lr_d = lr_d; self.different_platforms = different_platforms
        self.decode_axf, self.encode_slice, self.direct_counts, self.count_dist = decode_axf, encode_slice, direct_counts, count_dist
        self.gan_pairs = gan_pairs; self.beta_norm_slices = beta_norm_slices; self.seed = seed
        self.margin_warmup_step = 100
        self.margin = 5.0 if use_margin else 50.0
        self.n_spot_list = [a.shape[0] for a in adata_st_list]
        self.shared_hvgs = adata_st_list[0].var.index
        self.neighbor_idx_list = [a.obsm['neighbor_idx'] if 'neighbor_idx' in a.obsm else None for a in adata_st_list]

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.net = IntegrationNet_LGCN(hidden_dims=self.hidden_dims, n_features_lgcn=self.n_features_lgcn, n_hvgs=self.n_hvgs,
                                       n_factors=n_spatial_factors, n_slices=self.n_slices, slice_emb_dim=self.slice_emb_dim,
                                       different_platforms=different_platforms, decode_axf=decode_axf, encode_slice=encode_slice,
                                       direct_counts=direct_counts).to(self.device)
        if count_dist == 'nb':   # per-gene log dispersion; part of the net so it is optimised and checkpointed
            self.net.log_theta = nn.Parameter(torch.full((self.n_hvgs,), 2.0, device=self.device))
        self.optimizer_net = optim.Adamax(list(self.net.parameters()), lr=self.lr, weight_decay=self.weight_decay)

        self.coef_fgw, self.fgw_mode, self.fgw_tau, self.fgw_alpha = coef_fgw, fgw_mode, fgw_tau, fgw_alpha
        self.fgw_warmup, self.fgw_every, self.fgw_S = fgw_warmup, fgw_every, fgw_spots_per_block
        self.fgw = None
        if fgw_spec is not None and coef_fgw > 0:
            self._fgw_setup(adata_st_list, fgw_spec)

        # discriminator pairs: (i, j) = section i is "real", section j is "fake"
        if gan_pairs == 'chain':
            self.d_pairs = [(i, i + 1) for i in range(self.n_slices - 1)]
        else:
            self.d_pairs = [(i, j) for i in range(self.n_slices) for j in range(i + 1, self.n_slices)]
        self.n_discriminator = len(self.d_pairs)
        self.discriminator = {}; d_params = []
        for d_i in range(self.n_discriminator):
            self.discriminator[d_i] = DiscriminatorNet(n_input=self.hidden_dims[-1], n_hidden=self.n_hidden_d).to(self.device)
            d_params += list(self.discriminator[d_i].parameters())
        self.optimizer_d = optim.Adam(d_params, lr=lr_d, weight_decay=self.weight_decay_d) if d_params else None

    # ------------------------------------------------------------------ training
    def _sample(self, adata_st_list):
        """One minibatch: batch_size spots drawn with replacement from every section."""
        nf, cnt, lib, lab, idx = {}, {}, {}, {}, {}
        for i in range(self.n_slices):
            ib = np.random.choice(np.arange(self.n_spot_list[i]), size=self.batch_size); idx[i] = ib
            a = adata_st_list[i]
            nf[i] = torch.from_numpy(a.obsm['node_features'][ib, :]).float().to(self.device)
            cnt[i] = torch.from_numpy(np.array(a.obsm['count'][ib, :])).float().to(self.device)
            lib[i] = torch.from_numpy(np.array(a.obs['library_size'].values.reshape(-1, 1))[ib, :]).float().to(self.device)
            lab[i] = torch.from_numpy(np.array(a.obs['slice'].values))[ib].long().to(self.device)
        return nf, cnt, lib, lab, idx

    def _disc_terms(self, Zs, step, fake_only):
        """Sum over discriminator pairs of the logistic losses (clamped logits after the warm-up). With all C(S,2) pairs the sum
        is rescaled to the magnitude of the S-1 chain (/ n_pairs * (S-1)), so coef_gan needs no retuning."""
        clamp = (lambda s: s) if step <= self.margin_warmup_step else (lambda s: torch.clamp(s, -self.margin, self.margin))
        tot = 0.
        for d_i, (i, j) in enumerate(self.d_pairs):
            D = self.discriminator[d_i]
            if fake_only:   # generator: make section j look like section i
                tot = tot + torch.mean(torch.log(1 + torch.exp(-clamp(D(Zs[j])))))
            else:           # discriminator: i real, j fake
                tot = tot + torch.mean(torch.log(1 + torch.exp(-clamp(D(Zs[i]))))) + torch.mean(torch.log(1 + torch.exp(clamp(D(Zs[j])))))
        if self.gan_pairs == 'all':
            tot = tot / self.n_discriminator * (self.n_slices - 1)
        return tot

    def _count_nll(self, cnt, lib, log_lam):
        """Negative log-likelihood of the counts (constant terms in x dropped)."""
        if self.count_dist == 'nb':
            log_mu = torch.log(lib + 1e-6) + log_lam
            lt = self.net.log_theta.clamp(-6, 8); th = torch.exp(lt); log_tm = torch.logaddexp(lt.expand_as(log_mu), log_mu)
            ll = torch.lgamma(cnt + th) - torch.lgamma(th) + th * (lt - log_tm) + cnt * (log_mu - log_tm)
            return -torch.mean(torch.sum(ll, axis=1))
        return -torch.mean(torch.sum(cnt * (torch.log(lib + 1e-6) + log_lam) - lib * torch.exp(log_lam), axis=1))

    def train(self, adata_st_list, loss_log_path=None, loss_log_interval=10, checkpoint_steps=(), checkpoint_dir=None,
              checkpoint_eval_batch_size=4096, resume_ckpt=None):
        """Train for n_training_steps. Writes a CSV loss log (every loss_log_interval steps) and checkpoints
        ckpt_step<N>.pt (net + optimiser) with latent_step<N>.npz (Z, beta of all spots) at the requested steps."""
        self.net.train()
        for d in self.discriminator.values(): d.train()
        start_step = 0
        if resume_ckpt is not None:
            ck = torch.load(resume_ckpt, map_location=self.device, weights_only=False)
            self.net.load_state_dict(ck['net']); self.optimizer_net.load_state_dict(ck['optimizer']); start_step = int(ck['step'])
            print('Resumed from %s at step %d' % (resume_ckpt, start_step))
        if loss_log_path is not None:
            open(loss_log_path, 'w').write(','.join(LOSS_COLUMNS) + '\n')
        if checkpoint_dir is not None: os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_steps = set(checkpoint_steps)
        G = self.n_hvgs

        for step in tqdm(range(start_step, self.n_training_steps)):
            nf, cnt, lib, lab, idx = self._sample(adata_st_list)
            Zs, betas, alphas, recs, recs_axf, basis_val, gammas = self.net(nf, lab)

            # 1. discriminator step (on detached-by-retain graph, as in INSPIRE)
            loss_d = 0.
            if self.optimizer_d is not None:
                self.optimizer_d.zero_grad()
                loss_d = self._disc_terms(Zs, step, fake_only=False)
                (self.coef_gan * loss_d).backward(retain_graph=True)
                self.optimizer_d.step()

            # 2. auto-encoder, count likelihood, geometry
            features_loss = recon_loss = geom_loss = 0.
            for i in range(self.n_slices):
                features_loss = features_loss + torch.mean(torch.sqrt(torch.sum(torch.pow(nf[i][:, :G] - recs[i], 2), axis=1)))
                if self.decode_axf:
                    features_loss = features_loss + torch.mean(torch.sqrt(torch.sum(torch.pow(nf[i][:, G:] - recs_axf[i], 2), axis=1)))
                if self.direct_counts:
                    log_lam = F.log_softmax(self.net.count_logits(Zs[i], self.net.slice_emb(lab[i])), dim=1) + gammas[lab[i]]
                else:
                    log_lam = torch.log(torch.matmul(betas[i], basis_val) + 1e-6) + gammas[lab[i]]
                recon_loss = recon_loss + self._count_nll(cnt[i], lib[i], log_lam)
                Zn = F.normalize(Zs[i], p=2); Xn = F.normalize(nf[i][:, :G], p=2)
                geom_loss = geom_loss + torch.mean(torch.sum(torch.pow(Zn @ Zn.T - Xn @ Xn.T, 2), axis=1))

            # 3. Dirichlet prior on the mean factor usage
            p_aver = sum(torch.sum(betas[i], axis=0) for i in range(self.n_slices))
            p_aver = p_aver / (self.batch_size * (self.beta_norm_slices or self.n_slices))
            beta_loss = - torch.sum((5.0 - 1.) * torch.log(p_aver + 1e-6))

            # 4. patch variance of beta (spot + its 8 nearest grid neighbours), per section
            patch_var_loss = 0.
            if self.coef_patch_var > 0:
                for i in range(self.n_slices):
                    if self.neighbor_idx_list[i] is None: continue
                    nbr = self.neighbor_idx_list[i][idx[i]]; K = nbr.shape[1]
                    nbr_feats = torch.from_numpy(adata_st_list[i].obsm['node_features'][nbr.reshape(-1), :]).float().to(self.device)
                    nbr_lab = torch.full((nbr_feats.shape[0],), int(adata_st_list[i].obs['slice'].values[0]), dtype=torch.long, device=self.device)
                    b_nbr = F.softmax(self.net.deconv_beta_layer(F.elu(self.net.encoder(nbr_feats, nbr_lab))), dim=1).view(self.batch_size, K, -1)
                    patch_beta = (betas[i] + b_nbr.sum(dim=1)) / (K + 1)
                    patch_var_loss = patch_var_loss - torch.sum(torch.log(torch.var(patch_beta, dim=0, unbiased=False) + 1e-6))

            # 5. generator side of the adversarial alignment
            gan_loss = self._disc_terms(Zs, step, fake_only=True) if self.n_discriminator else 0.

            # 6. FGW column-anchor alignment (joint mode: recompute the coupling on schedule)
            fgw_loss = 0.
            if self.fgw is not None:
                if self.fgw_mode == 'joint' and step >= self.fgw_warmup and (step - self.fgw_warmup) % self.fgw_every == 0:
                    self._fgw_recompute(step)
                fgw_loss = self._fgw_loss()
                if not step % self.step_interval: print('FGW step: %d, fgw_loss: %.4f' % (step, fgw_loss.item()))

            self.optimizer_net.zero_grad()
            loss_total = (self.coef_fe * features_loss + self.coef_recon * recon_loss + self.coef_geom * geom_loss + self.coef_beta * beta_loss
                          + self.coef_patch_var * patch_var_loss + self.coef_gan * gan_loss)
            if self.fgw is not None: loss_total = loss_total + self.coef_fgw * fgw_loss
            loss_total.backward()
            self.optimizer_net.step()

            vals = [step, loss_total, recon_loss, features_loss, geom_loss, beta_loss, patch_var_loss, gan_loss, loss_d, fgw_loss]
            if not step % self.step_interval:
                print('Step: ' + ', '.join('%s: %.4f' % (k, _val(v)) for k, v in zip(LOSS_COLUMNS[1:], vals[1:])))
            if loss_log_path is not None and not step % loss_log_interval:
                open(loss_log_path, 'a').write('%d,' % step + ','.join('%.6f' % _val(v) for v in vals[1:]) + '\n')
            if checkpoint_dir is not None and (step + 1) in checkpoint_steps:
                n_done = step + 1
                torch.save({'step': n_done, 'net': self.net.state_dict(), 'optimizer': self.optimizer_net.state_dict()},
                           os.path.join(checkpoint_dir, 'ckpt_step%d.pt' % n_done))
                Z_ck, beta_ck = self.encode_minibatch(adata_st_list, checkpoint_eval_batch_size)
                np.savez_compressed(os.path.join(checkpoint_dir, 'latent_step%d.npz' % n_done), Z=Z_ck, beta=beta_ck)
                self.net.train()

    # ------------------------------------------------------------------ FGW column anchors
    def _fgw_setup(self, adata_st_list, spec):
        """Map the spec's spot ids to positions, preload node spots' features on the device, build per-layer targets."""
        layers = list(spec['layers']); sid = {str(a.obs['slice_id'].values[0]): i for i, a in enumerate(adata_st_list)}
        self.fgw = {'layers': layers, 'slices': {}, 'pairs': [], 'history': []}
        for sf, sd in spec['slices'].items():
            if sf not in sid: continue
            i = sid[sf]; a = adata_st_list[i]; names = pd.Index(a.obs_names.astype(str))
            allpos, blocks, found, total = [], [], 0, 0
            for bl in sd['blocks']:                     # bl: {layer: spot ids} of one column (1D) or one node (2D, layer 'all')
                d = {}
                for l, ids in bl.items():
                    p = names.get_indexer(ids); total += len(p); p = p[p >= 0]; found += len(p)
                    if len(p) == 0: continue
                    d[l] = np.arange(len(allpos), len(allpos) + len(p)); allpos.extend(p.tolist())
                blocks.append(d)
            print('FGW setup: %s columns %d, spots found %d/%d' % (sf, len(blocks), found, total))
            feats = torch.from_numpy(a.obsm['node_features'][np.array(allpos)]).float().to(self.device)
            lab = torch.full((len(allpos),), int(a.obs['slice'].values[0]), dtype=torch.long, device=self.device)
            per_layer = {}
            for l in layers:                            # padded index matrix of the node spots, per layer
                cols = [c for c, d in enumerate(blocks) if l in d]
                if not cols: continue
                L = max(len(blocks[c][l]) for c in cols)
                ix = np.zeros((len(cols), L), np.int64); ln = np.zeros(len(cols), np.int64)
                for r, c in enumerate(cols):
                    v = blocks[c][l]; ix[r, :len(v)] = v; ln[r] = len(v)
                per_layer[l] = dict(cols=np.array(cols), idx=torch.from_numpy(ix).to(self.device), len=torch.from_numpy(ln).to(self.device))
            self.fgw['slices'][sf] = dict(i=i, feats=feats, lab=lab, blocks=blocks, per_layer=per_layer, n_cols=len(blocks), arc=np.asarray(sd['arc'], float))
        for pr in spec.get('pairs', []):
            if pr['a'] not in self.fgw['slices'] or pr['b'] not in self.fgw['slices']: continue
            P = dict(a=pr['a'], b=pr['b'], CA=pr['CA'], CB=pr['CB'], p=pr['p'], q=pr['q'], T0=np.asarray(pr['T'], np.float64))
            self._fgw_set_T(P, P['T0']); self.fgw['pairs'].append(P)
        print('FGW setup: %d slice pairs, mode %s' % (len(self.fgw['pairs']), self.fgw_mode))

    def _fgw_set_T(self, P, T):
        """Row-normalised soft targets A->B and B->A per layer (restricted to the nodes present in that layer)."""
        P['T'] = T; P['targets'] = {}
        A, B = self.fgw['slices'][P['a']], self.fgw['slices'][P['b']]
        for l in self.fgw['layers']:
            if l not in A['per_layer'] or l not in B['per_layer']: continue
            ra, rb = A['per_layer'][l]['cols'], B['per_layer'][l]['cols']; sub = T[np.ix_(ra, rb)]
            tab = sub / np.maximum(sub.sum(1, keepdims=True), 1e-12); tba = sub.T / np.maximum(sub.T.sum(1, keepdims=True), 1e-12)
            P['targets'][l] = (torch.from_numpy(tab).float().to(self.device), torch.from_numpy(sub.sum(1) > 0).to(self.device),
                               torch.from_numpy(tba).float().to(self.device), torch.from_numpy(sub.sum(0) > 0).to(self.device))

    def _fgw_block_means(self, sf, sample=True):
        """Latent mean of every node: fgw_spots_per_block sampled spots (training) or all spots (no grad, recomputation)."""
        S = self.fgw['slices'][sf]; out = {}
        for l, pl in S['per_layer'].items():
            if sample:
                r = (torch.rand(len(pl['cols']), self.fgw_S, device=self.device) * pl['len'][:, None]).long()
                pos = torch.gather(pl['idx'], 1, r).reshape(-1)
                z = self.net.encoder(S['feats'][pos], S['lab'][pos]).reshape(len(pl['cols']), self.fgw_S, -1).mean(1)
            else:
                z = torch.stack([self.net.encoder(S['feats'][c[:n]], S['lab'][c[:n]]).mean(0) for c, n in zip(pl['idx'], pl['len'])])
            out[l] = z
        return out

    def _fgw_loss(self):
        """Cross-entropy between the FGW coupling (targets) and the softmax of node cosine similarities, both directions."""
        means = {sf: self._fgw_block_means(sf) for sf in self.fgw['slices']}
        total, n = 0., 0
        for P in self.fgw['pairs']:
            for l, (tab, ma, tba, mb) in P['targets'].items():
                za = F.normalize(means[P['a']][l], dim=1); zb = F.normalize(means[P['b']][l], dim=1)
                sim = za @ zb.t() / self.fgw_tau
                lab_ = -(tab * F.log_softmax(sim, dim=1)).sum(1)[ma].mean()
                lba = -(tba * F.log_softmax(sim.t(), dim=1)).sum(1)[mb].mean()
                total = total + 0.5 * (lab_ + lba); n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def _fgw_recompute(self, step):
        """Joint mode: new coupling from the current latent. Node profiles = mean over layers of the node latents
        (z-scored within the section); balanced FGW with the spec's structure matrices."""
        import ot
        was = self.net.training; self.net.eval(); prof = {}
        for sf, S in self.fgw['slices'].items():
            m = self._fgw_block_means(sf, sample=False); d_ = next(iter(m.values())).shape[1]
            pr = torch.zeros(S['n_cols'], d_, device=self.device); cnt = torch.zeros(S['n_cols'], 1, device=self.device)
            for l, z in m.items():
                c = torch.from_numpy(S['per_layer'][l]['cols']).to(self.device); pr[c] += z; cnt[c] += 1
            pr = (pr / cnt.clamp(min=1)).cpu().numpy(); prof[sf] = (pr - pr.mean(0)) / (pr.std(0) + 1e-9)
        for P in self.fgw['pairs']:
            M = ot.dist(prof[P['a']], prof[P['b']]); M /= M.max()
            T = np.asarray(ot.gromov.fused_gromov_wasserstein(M, P['CA'], P['CB'], P['p'], P['q'], loss_fun='square_loss', alpha=self.fgw_alpha))
            changed = float(np.mean(T.argmax(1) != P['T'].argmax(1))); vs0 = float(np.mean(T.argmax(1) != P['T0'].argmax(1)))
            self.fgw['history'].append(dict(step=step, a=P['a'], b=P['b'], changed_vs_prev=changed, changed_vs_static=vs0))
            print('FGW recompute step %d %s|%s: argmax changed vs previous %.3f, vs static %.3f' % (step, P['a'], P['b'], changed, vs0))
            self._fgw_set_T(P, T)
        if was: self.net.train()

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def encode_minibatch(self, adata_st_list, batch_size):
        """Deterministic Z and beta of all spots (sections concatenated in order); restores train mode."""
        self.net.eval(); Zs, Bs = [], []
        for i in range(self.n_slices):
            a = adata_st_list[i]
            for b0 in range(0, a.shape[0], batch_size):
                nf = torch.from_numpy(a.obsm['node_features'][b0:b0 + batch_size, :]).float().to(self.device)
                lab = torch.from_numpy(np.array(a.obs['slice'].values[b0:b0 + batch_size])).long().to(self.device)
                Z = self.net.encoder(nf, lab); beta, _ = self.net.deconvolutioner(Z, self.net.slice_emb(lab))
                Zs.append(Z.cpu().numpy()); Bs.append(beta.cpu().numpy())
        self.net.train()
        return np.concatenate(Zs), np.concatenate(Bs)

    def eval_minibatch(self, adata_st_list, adata_full, batch_size):
        """Z into adata_full.obsm['latent'], beta into obs ('Proportion of spatial factor k'); returns (adata_full, basis)."""
        Z, beta = self.encode_minibatch(adata_st_list, batch_size); self.net.eval()
        assert Z.shape[0] == adata_full.shape[0]
        dec = pd.DataFrame(beta, columns=['Proportion of spatial factor ' + str(j + 1) for j in range(beta.shape[1])], index=adata_full.obs.index)
        adata_full.obs = adata_full.obs.join(dec); adata_full.obsm['latent'] = Z
        basis = self.net.get_basis().detach().cpu().numpy()
        return adata_full, pd.DataFrame(basis, columns=self.shared_hvgs)
