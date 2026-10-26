#!/bin/bash
# Step 03: FGW anchor specs of the series: 1D (columns) and 2D (column x depth bin), alpha 0.7, ring sections circular,
# neighbouring sections up to 2 apart (CPU, ~5 min)
source "$(dirname "$0")/common.sh"
$PY -m ottr.fgw.spec2d --ds $DS --alpha 0.7 --circ --pairs_k 2 > logs/03_spec2d.log 2>&1 || { echo "spec2d FAILED"; exit 1; }
$PY -m ottr.fgw.spec1d --ds $DS --alpha 0.7 --circ --pairs_k 2 > logs/03_spec1d.log 2>&1 || { echo "spec1d FAILED"; exit 1; }
echo "step 03 done"
