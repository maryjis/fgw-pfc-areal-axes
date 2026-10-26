"""Shared colours and helpers of the figures."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import cfg, ensure

AREA_COL = {'FPC': '#7f7f7f', 'DLPFC': '#1f5fb4', 'Premotor': '#9467bd', 'VLPFC': '#2ca02c', 'OPFC': '#e6750d', 'MPFC': '#d62728', 'ACC': '#8c564b'}
DEV_COL = {'mPFC': '#d62728', 'oPFC': '#e6750d', 'vlPFC': '#2ca02c', 'dlPFC': '#1f5fb4'}
LAYER_COL = {'L1': '#1c5cab', 'L2': '#86b6ef', 'L3': '#eda100', 'L4': '#1baf7a', 'L5': '#e87ba4', 'L6': '#c2461a'}
AGE_COL = {'adult': '#14213d', 'P0': '#7a3d9a', 'P32': '#1d4ed8', '3M': '#0f766e', '7M': '#c2410c'}
MODEL_COL = {'smoothed_pca': '#8a8f99', 'inspire_gan': '#e87ba4', 'fgw1d_factors': '#7a3d9a', 'fgw2d_factors': '#2a78d6',
             'fgw2d_poisson': '#eb6834', 'fgw2d_nb': '#1baf7a', 'classic': '#b8860b'}
COMP_COL = ['#1d4ed8', '#7a3d9a', '#c2410c', '#0f766e', '#8a8f99']
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': '#b5b4ae', 'axes.labelcolor': '#333333', 'xtick.color': '#444444', 'ytick.color': '#444444'})


def reading_colour(rd):
    """Colour of an axis title by its reading: SA blue, areal / regional purple, section effect orange, depth / other grey."""
    return ('#1d4ed8' if rd.startswith('SA') else '#7a3d9a' if (rd.startswith('region') or rd.startswith('areal') or rd.startswith('rostro')) else
            '#c2410c' if rd.startswith('section') else '#6b7280' if (rd.startswith('depth') or rd.startswith('not') or rd.startswith('arch')) else '#14213d')


def save(fig, name, dpi=70):
    """Save into results/pictures/<name>.png and close."""
    p = os.path.join(ensure(cfg().pictures()), f'{name}.png'); fig.savefig(p, dpi=dpi, facecolor='white'); plt.close(fig); print('saved', p, flush=True)
    return p
