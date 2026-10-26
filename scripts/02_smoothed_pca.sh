#!/bin/bash
# Step 02: smoothed PCA r 1.5 (the FGW anchor space and the training-free reference); with --select also the radius curve and
# the radius choice (evaluates 6 radii, ~60 min)
source "$(dirname "$0")/common.sh"
$PY -m ottr.smoothed_pca.curve --ds $DS --radii 1.5 > logs/02_curve.log 2>&1 || { echo "curve FAILED"; exit 1; }
[ "$1" == "--select" ] && { $PY -m ottr.smoothed_pca.select --ds $DS > logs/02_select.log 2>&1 || echo "radius selection FAILED"; }
echo "step 02 done"
