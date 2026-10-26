#!/bin/bash
# Level-1 reproduction: run every deterministic analysis of the new code on the previous project's stored embeddings
# (reference workspace, configs/reference.yaml); tests/test_reference.py compares the outputs with the stored results.
# ~90 min on 32 cores.  usage: bash scripts/tools/run_reference.sh
set -e; cd "$(dirname "$0")/../.."
PY=$(awk '$1=="analysis:"{print $2}' configs/default.yaml)     # analysis environment (numerics of the stored results)
PYT=$(awk '$1=="train:"{print $2}' configs/default.yaml)       # Leiden (evaluate, nmi_within) ran in the training environment (scanpy 1.11)
export OTTR_CONFIG=configs/reference.yaml PYTHONPATH=src OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
$PY scripts/tools/reference_workspace.py
F2=fgw2d_joint_a07c_c770_k2_s20000; P2=fgw2d_joint_a07c_c770_k2_direct_s20000; N2=fgw2d_joint_a07c_c770_k2_directnb_s20000
G=encslice_gan1_s20000; S=curve_r1.5_centred; F1=fgw_joint_a07c_c770_k2_s20000
for e in $S $G $F2; do $PYT -m ottr.metrics.evaluate pfcserL $e --leiden; done                      # one process per embedding
$PYT -m ottr.metrics.nmi_within pfcserL $S,$G,$F2
$PY -m ottr.metrics.series "smoothed PCA:$S,INSPIRE 20000:$G,FGW 2D factors 20000:$F2" --out work_ref/results/tables/series_metrics.csv --rank_out work_ref/results/tables/series_rank.csv
$PY -m ottr.areal.multi_axes "smoothed PCA:$S,INSPIRE:$G,FGW 1D:$F1,FGW 2D:$F2"
$PY -m ottr.areal.multi_axes "FGW 2D:$P2" --tag _d2d20k
$PY -m ottr.areal.multi_axes "FGW 2D:$N2" --tag _d2dnb20k
$PY -m ottr.areal.spot_axes "FGW 2D:$F2" --tag _k2_20k
$PY -m ottr.areal.spot_axes "FGW 2D:$P2" --tag _d2d20k
$PY -m ottr.areal.spot_axes "FGW 2D:$N2" --tag _d2dnb20k
$PY -m ottr.atlas.frontal_pca
$PY -m ottr.atlas.flatmap assign
$PY -m ottr.atlas.flatmap project --models "FGW 2D" --embs $F2 --spot_tag _k2_20k --out flatmaps20000
$PY -m ottr.atlas.flatmap project --tag _d2d20k --models "FGW 2D" --embs $P2 --spot_tag _d2d20k --out flatmaps_d2d20k
$PY -m ottr.atlas.flatmap project --tag _d2dnb20k --models "FGW 2D" --embs $N2 --spot_tag _d2dnb20k --out flatmaps_d2dnb20k
$PY -c "from ottr.atlas.match import match; match([('factors','flatmaps20000','FGW 2D'),('no factors · Poisson','flatmaps_d2d20k','FGW 2D'),('no factors · NB','flatmaps_d2dnb20k','FGW 2D')])"
$PY -c "
from ottr.areal.development import AxesModel, project, conservation, age_areal, age_metrics
for M in [AxesModel('factors', 'FGW 2D', '', '$F2', 'adultenc_fgw2d20k', ''), AxesModel('poisson', 'FGW 2D', '_d2d20k', '$P2', 'adultencNF_fgw2d20k', '_nf20k'),
          AxesModel('nb', 'FGW 2D', '_d2dnb20k', '$N2', 'adultencNB_fgw2d20k', '_nb20k')]:
    for ds in ['devP0', 'devP32', 'dev3M', 'dev7M']: project(M, ds)
    conservation(M); age_areal(M, 'median'); age_metrics(M)
    if M.key == 'factors': age_areal(M, 'mean')
"
echo REFERENCE_DONE
