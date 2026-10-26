#!/bin/bash
# Step 09: adult axes in the developing PFC (P0-7M encoded by the adult models in step 05) (CPU, ~30 min)
source "$(dirname "$0")/common.sh"
$PY -c "
from ottr.registry import FGW2D, axes_model
from ottr.areal.development import project, conservation, age_areal, age_metrics
for k, _, _, _ in FGW2D:
    M = axes_model(k)
    for ds in ['devP0', 'devP32', 'dev3M', 'dev7M']: project(M, ds)
    conservation(M); age_areal(M, 'median'); age_metrics(M)
" > logs/09_dev.log 2>&1 || { echo "development FAILED"; exit 1; }
echo "step 09 done"
