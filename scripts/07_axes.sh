#!/bin/bash
# Step 07: axes of the latents by four methods incl. the areal-axis model (multi_axes) and the spot-level model (CPU, ~60 min)
source "$(dirname "$0")/common.sh"
$PY -c "
from ottr.registry import AXES_RUNS, MODELS
from ottr.areal import multi_axes, spot_axes
for tag, items in AXES_RUNS.items():
    multi_axes.run([(lab, MODELS[k][1]) for lab, k in items], tag=tag)
    for lab, k in items:
        if lab == 'FGW 2D': spot_axes.run([(lab, MODELS[k][1])], tag=tag)
" > logs/07_axes.log 2>&1 || { echo "axes FAILED"; exit 1; }
echo "step 07 done"
