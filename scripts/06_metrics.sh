#!/bin/bash
# Step 06: metrics of every model (evaluate + Leiden NMI in the training environment, one process per model), series metrics,
# ranking, reconstruction R2 (CPU, ~90 min)
source "$(dirname "$0")/common.sh"
EMBS=$($PY -c "from ottr.registry import MODELS, RANK_SET; print(' '.join(MODELS[k][1] for k in RANK_SET))")
for e in $EMBS; do $PYT -m ottr.metrics.evaluate $DS $e --leiden > logs/06_eval_$e.log 2>&1 || echo "evaluate $e FAILED"; done
$PYT -m ottr.metrics.nmi_within $DS $(echo $EMBS | tr ' ' ',') > logs/06_nmi.log 2>&1 || echo "nmi FAILED"
ITEMS=$($PY -c "from ottr.registry import MODELS, RANK_SET; print(','.join(f'{MODELS[k][0]}:{MODELS[k][1]}' for k in RANK_SET))")
$PY -m ottr.metrics.series "$ITEMS" > logs/06_series.log 2>&1 || echo "series metrics FAILED"
$PY -c "
from ottr.metrics.recon_r2 import run; from ottr.registry import MODELS, FGW2D; from ottr.config import cfg; import os
run([(MODELS[k][0], os.path.join(cfg().work, 'recon', k)) for k, _, _, _ in FGW2D if os.path.exists(os.path.join(cfg().work, 'recon', k + '_features.npz'))])
" > logs/06_recon.log 2>&1 || echo "recon R2 FAILED"
echo "step 06 done"
