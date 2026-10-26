"""The models of the benchmark and how the later steps refer to them (one place for names, labels and groupings)."""
from .areal.development import AxesModel

# key -> (display label, embedding name in work/embeddings/<series>/, trained? (results/models/<key>/))
MODELS = {
    'smoothed_pca': ('smoothed PCA', 'curve_r1.5_centred', False),
    'inspire_gan': ('INSPIRE (GAN)', 'inspire_gan', True),
    'fgw1d_factors': ('FGW 1D · factors', 'fgw1d_factors', True),
    'fgw2d_factors': ('FGW 2D · factors', 'fgw2d_factors', True),
    'fgw2d_poisson': ('FGW 2D · no factors · Poisson', 'fgw2d_poisson', True),
    'fgw2d_nb': ('FGW 2D · no factors · NB', 'fgw2d_nb', True),
}
RANK_SET = ['smoothed_pca', 'inspire_gan', 'fgw1d_factors', 'fgw2d_factors', 'fgw2d_poisson', 'fgw2d_nb']

# multi_axes runs (tag -> [(label inside the run, model key)]); the main run holds the four-way comparison as in the
# previous project, the no-factor models have their own runs (the random state is shared within a run)
AXES_RUNS = {'': [('smoothed PCA', 'smoothed_pca'), ('INSPIRE', 'inspire_gan'), ('FGW 1D', 'fgw1d_factors'), ('FGW 2D', 'fgw2d_factors')],
             '_poisson': [('FGW 2D', 'fgw2d_poisson')],
             '_nb': [('FGW 2D', 'fgw2d_nb')]}
# FGW 2D models carried to the atlas comparison, the development analysis and the 3D pages
FGW2D = [('fgw2d_factors', '', 'factors', 'With spatial factors'), ('fgw2d_poisson', '_poisson', 'no factors · Poisson', 'No factors · Poisson'),
         ('fgw2d_nb', '_nb', 'no factors · NB', 'No factors · neg. binomial')]


def emb(key):
    return MODELS[key][1]


def dev_emb(key):
    """Embedding of the young data encoded by the adult model `key`."""
    return f'adultenc_{key}'


def axes_model(key):
    """AxesModel of an FGW 2D model (multi_axes label 'FGW 2D' in its run)."""
    tag = {k: t for k, t, _, _ in FGW2D}[key]
    return AxesModel(key=key, label='FGW 2D', axes_tag=tag, adult_emb=emb(key), dev_emb=dev_emb(key), tag=f'_{key}')
