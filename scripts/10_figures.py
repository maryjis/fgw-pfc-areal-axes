"""Step 10: pictures (results/pictures) and the 3D pages (results/3d_model) from the tables of steps 06-09."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from ottr.config import cfg
from ottr.registry import MODELS, RANK_SET, FGW2D, axes_model
from ottr.viz import summary, umap_series, axes_maps, dev, brain3d
c = cfg(); only = set(sys.argv[1:])        # optional subset: rank umap axes atlas dev losses 3d
want = lambda x: not only or x in only
if want('rank'):
    summary.ranking()
    if os.path.exists(c.tables(f"smoothing_radius_{c.series['name']}.csv")): summary.smoothing_radius()
if want('losses'):
    summary.losses([(k, MODELS[k][0], os.path.join(c.models(k), 'loss_log.csv')) for k in RANK_SET if MODELS[k][2]])
    if os.path.exists(c.tables('recon_r2.csv')): summary.recon()
if want('umap'): umap_series.plot([(MODELS[k][1], MODELS[k][0]) for k in RANK_SET])
if want('axes'):
    for k, tag, _, title in FGW2D:
        axes_maps.tissue(tag, 'FGW 2D', tag, f'{MODELS[k][0]}', k); axes_maps.regions(tag, 'FGW 2D', MODELS[k][0], k)
if want('atlas'): summary.atlas_heatmap()
if want('dev'):
    for k, _, _, _ in FGW2D:
        M = axes_model(k); dev.spatial(M, MODELS[k][0]); dev.conservation(M, MODELS[k][0]); dev.age(M, MODELS[k][0])
if want('3d'):
    A = brain3d.atlas_data([dict(key=k, label=title, axes_tag=tag, axes_label='FGW 2D', emb=MODELS[k][1],
                                 spot_tag=tag, match_label=ml) for k, tag, ml, title in FGW2D])
    brain3d.build_page('brain3d_atlas.html', A, 'adult_atlas_pcs_vs_axes_3d.html')
    D = brain3d.development_data([(axes_model(k), title, ml) for k, _, ml, title in FGW2D], A)
    brain3d.build_page('brain3d_development.html', D, 'development_axes_3d.html')
