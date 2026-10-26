"""Neural networks of INSPIRE (LGCN variant), extended for this project.

Original: Zhao et al., INSPIRE (github.com/jiazhao97/INSPIRE, INSPIRE/networks.py, IntegrationNet_LGCN, DiscriminatorNet).

IntegrationNet: an encoder maps the node features of a spot, x = [log1p CP10k of the genes, the same smoothed over the spatial
graph (A x)], to a latent Z (32 dims). Two heads are trained from Z (+ a 4-dim embedding of the section id):
  * decoder        Z -> reconstruction of the spot's own log expression (feature loss, "denoising auto-encoder" part)
  * deconvolutioner Z -> beta (softmax over K spatial factors); count rate = beta @ softmax(factors) (gene programmes)
                    times the library size times exp(gamma[section, gene]) (Poisson likelihood of the raw counts)
Extensions (all off by default, so the original network is recovered):
  * decode_axf     second decoder that reconstructs the graph-smoothed half of the features (A x) as well
  * encode_slice   the one-hot section id is appended to the encoder input (scVI-style covariate)
  * direct_counts  third decoder Z (+ section embedding) -> gene logits; count rate = library * softmax(logits) * exp(gamma)
                   i.e. the counts are decoded WITHOUT the spatial-factor factorisation ("no factors" models)
Parameter names (encoder_layers.*, decoder_layers.*, ...) are those of the project's benchmark copy, so its checkpoints load.
The layers are created in the same order as in the original code, so with the extensions off and the same seed the initial
weights equal those of the original package (tests/test_inspire_classic.py)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """Linear layer with the INSPIRE initialisation (Xavier uniform weights or zeros, zero bias)."""

    def __init__(self, c_in, c_out, zero_init=False):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out)
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, x):
        return self.linear(x)


class DiscriminatorNet(nn.Module):
    """Section discriminator of the adversarial alignment (2 hidden ReLU layers, score clamped to +-50)."""

    def __init__(self, n_input, n_hidden):
        super().__init__()
        self.discriminator_layer1 = DenseLayer(n_input, n_hidden)
        self.discriminator_layer2 = DenseLayer(n_hidden, n_hidden)
        self.discriminator_layer3 = DenseLayer(n_hidden, 1)

    def forward(self, Z):
        H = F.relu(self.discriminator_layer1(Z))
        H = F.relu(self.discriminator_layer2(H))
        return torch.clamp(self.discriminator_layer3(H), min=-50.0, max=50.0)


def _mlp(dims):
    return nn.ModuleList([DenseLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])


def _run_mlp(layers, H):
    """ELU between layers, no activation after the last one."""
    for i, layer in enumerate(layers):
        H = layer(H)
        if i < len(layers) - 1:
            H = F.elu(H)
    return H


class IntegrationNet_LGCN(nn.Module):

    def __init__(self,
                 hidden_dims,            # hidden widths, the LAST one is the latent dim: [512, 32] = original INSPIRE
                 n_features_lgcn,        # encoder input: 2 x n_genes (log expression and its graph-smoothed copy)
                 n_hvgs,                 # number of genes
                 n_factors,              # number of spatial factors K
                 n_slices,               # number of sections
                 slice_emb_dim,          # dim of the learned section embedding (4 in INSPIRE)
                 different_platforms=False,   # INSPIRE option: beta also sees the section embedding
                 decode_axf=False,       # extension: also reconstruct the smoothed half of the features
                 encode_slice=False,     # extension: section one-hot appended to the encoder input
                 direct_counts=False,    # extension: decode counts directly from Z (no factors)
                 ):
        super().__init__()
        self.n_slices = n_slices
        self.different_platforms = different_platforms
        self.decode_axf = decode_axf
        self.encode_slice = encode_slice
        self.direct_counts = direct_counts
        self.latent_dim = hidden_dims[-1]
        # encoder: features (+ section one-hot) -> ... -> Z ; decoder(s): Z + section embedding -> ... -> genes
        self.encoder_layers = _mlp([n_features_lgcn + (n_slices if encode_slice else 0)] + list(hidden_dims))
        dec_dims = [self.latent_dim + slice_emb_dim] + list(hidden_dims[:-1])[::-1] + [n_hvgs]
        self.decoder_layers = _mlp(dec_dims)
        if decode_axf:
            self.decoder2_layers = _mlp(dec_dims)
        if direct_counts:
            self.decoder_cnt_layers = _mlp(dec_dims)
        # deconvolution: beta (spatial-factor proportions), factor x gene programmes, alpha (as in INSPIRE, unused in the loss)
        self.deconv_beta_layer = DenseLayer(self.latent_dim + (slice_emb_dim if different_platforms else 0), n_factors)
        self.factors = nn.Parameter(torch.Tensor(n_factors, n_hvgs).uniform_(-np.sqrt(6 / (n_factors + n_hvgs)), np.sqrt(6 / (n_factors + n_hvgs))))
        self.deconv_alpha_layer = DenseLayer(self.latent_dim + slice_emb_dim, 1, zero_init=True)
        self.gamma = nn.Parameter(torch.Tensor(n_slices, n_hvgs).zero_())   # per-section, per-gene log scale of the counts
        self.slice_emb = nn.Embedding(n_slices, slice_emb_dim)

    def forward(self, node_feats_dict, slice_label_dict):
        """Per section i: Z, beta, alpha, reconstructions; plus the gene basis and gamma."""
        Z, beta, alpha, rec, rec_axf = {}, {}, {}, {}, {}
        basis = self.get_basis()
        for i in range(self.n_slices):
            Z[i] = self.encoder(node_feats_dict[i], slice_label_dict[i])
            emb = self.slice_emb(slice_label_dict[i])
            beta[i], alpha[i] = self.deconvolutioner(Z[i], emb)
            alpha[i] = torch.clamp(alpha[i], -0.1, 0.1)
            rec[i] = self.decoder(Z[i], emb)
            if self.decode_axf:
                rec_axf[i] = self.decoder2(Z[i], emb)
        return Z, beta, alpha, rec, rec_axf, basis, self.gamma

    def encoder(self, node_feats, slice_label=None):
        H = node_feats
        if self.encode_slice:
            H = torch.cat((H, F.one_hot(slice_label.long().view(-1), self.n_slices).float()), axis=1)
        return _run_mlp(self.encoder_layers, H)

    def get_basis(self):
        """Gene programmes of the spatial factors: softmax over genes, K x n_genes."""
        return F.softmax(self.factors, dim=1)

    def deconvolutioner(self, Z, slice_label_emb):
        x = torch.cat((Z, slice_label_emb), axis=1) if self.different_platforms else Z
        beta = F.softmax(self.deconv_beta_layer(F.elu(x)), dim=1)
        alpha = self.deconv_alpha_layer(F.elu(torch.cat((Z, slice_label_emb), axis=1)))
        return beta, alpha

    def decoder(self, Z, slice_label_emb):
        return _run_mlp(self.decoder_layers, torch.cat((Z, slice_label_emb), axis=1))

    def decoder2(self, Z, slice_label_emb):
        return _run_mlp(self.decoder2_layers, torch.cat((Z, slice_label_emb), axis=1))

    def count_logits(self, Z, slice_label_emb):
        """Gene logits of the direct count decoder (direct_counts=True)."""
        return _run_mlp(self.decoder_cnt_layers, torch.cat((Z, slice_label_emb), axis=1))
