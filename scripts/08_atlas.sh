#!/bin/bash
# Step 08: frontal PCA of the atlas, our spots on the atlas flat map, correspondence of our axes with the atlas PCs (CPU, ~20 min)
source "$(dirname "$0")/common.sh"
$PY -m ottr.atlas.frontal_pca > logs/08_frontal.log 2>&1 || { echo "frontal PCA FAILED"; exit 1; }
$PY -m ottr.atlas.flatmap assign > logs/08_assign.log 2>&1 || { echo "assign FAILED"; exit 1; }
$PY -c "
from ottr.registry import FGW2D, MODELS
from ottr.atlas import flatmap, match
for k, tag, ml, _ in FGW2D: flatmap.project(['FGW 2D'], {'FGW 2D': MODELS[k][1]}, tag, tag, f'flatmap_{k}')
match.match([(ml, f'flatmap_{k}', 'FGW 2D') for k, tag, ml, _ in FGW2D])
" > logs/08_match.log 2>&1 || { echo "atlas match FAILED"; exit 1; }
echo "step 08 done"
