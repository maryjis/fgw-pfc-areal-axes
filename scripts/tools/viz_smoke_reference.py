"""Smoke test of the figure and 3D code on the reference workspace (stored embeddings of the previous project, outputs of
scripts/tools/run_reference.sh): every figure function and both 3D pages are built once.
usage: OTTR_CONFIG=configs/reference.yaml python scripts/tools/viz_smoke_reference.py  -> work_ref/results/{pictures,3d_model}"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from ottr.config import cfg
from ottr.areal.development import AxesModel
from ottr.viz import summary, umap_series, axes_maps, dev, brain3d

assert 'work_ref' in cfg().work, 'run with OTTR_CONFIG=configs/reference.yaml'
F2 = 'fgw2d_joint_a07c_c770_k2_s20000'; P2 = 'fgw2d_joint_a07c_c770_k2_direct_s20000'; N2 = 'fgw2d_joint_a07c_c770_k2_directnb_s20000'
summary.ranking(); summary.atlas_heatmap()
umap_series.plot([('curve_r1.5_centred', 'smoothed PCA'), (F2, 'FGW 2D · factors')], name='umap_smoke')
axes_maps.tissue('', 'FGW 2D', '_k2_20k', 'FGW 2D · factors', 'ref_factors'); axes_maps.regions('', 'FGW 2D', 'FGW 2D · factors', 'ref_factors')
M = AxesModel('ref_factors', 'FGW 2D', '', F2, 'adultenc_fgw2d20k', ''); dev.spatial(M, 'FGW 2D · factors'); dev.conservation(M, 'FGW 2D · factors'); dev.age(M, 'FGW 2D · factors')
MS = [('factors', '', '_k2_20k', F2, 'adultenc_fgw2d20k', '', 'factors', 'With spatial factors'),
      ('poisson', '_d2d20k', '_d2d20k', P2, 'adultencNF_fgw2d20k', '_nf20k', 'no factors · Poisson', 'No factors · Poisson'),
      ('nb', '_d2dnb20k', '_d2dnb20k', N2, 'adultencNB_fgw2d20k', '_nb20k', 'no factors · NB', 'No factors · neg. binomial')]
A = brain3d.atlas_data([dict(key=k, label=t, axes_tag=at, axes_label='FGW 2D', emb=e, spot_tag=st, match_label=ml) for k, at, st, e, de, dt, ml, t in MS])
brain3d.build_page('brain3d_atlas.html', A, 'adult_atlas_pcs_vs_axes_3d.html')
D = brain3d.development_data([(AxesModel(k, 'FGW 2D', at, e, de, dt), t, ml) for k, at, st, e, de, dt, ml, t in MS], A)
brain3d.build_page('brain3d_development.html', D, 'development_axes_3d.html')
print('viz smoke test done')
