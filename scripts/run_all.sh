#!/bin/bash
# The whole pipeline in order. Steps 04-05 need a GPU; 04 runs for hours (one model per GPU, ~8-20 h per model on shared GPUs).
# usage: bash scripts/run_all.sh [--from N]   (--dry-run: print the steps only)
cd "$(dirname "$0")/.."; FROM=0; DRY=0
for a in "$@"; do case $a in --from=*) FROM=${a#*=};; --dry-run) DRY=1;; esac; done
PYT=$(awk '$1=="train:"{print $2}' configs/default.yaml)
STEPS=("00 $PYT scripts/00_check.py" "01 bash scripts/01_prepare_data.sh" "02 bash scripts/02_smoothed_pca.sh --select" "03 bash scripts/03_fgw_anchors.sh"
       "04 $PYT scripts/04_train.py" "05 $PYT scripts/05_collect_encode.py" "06 bash scripts/06_metrics.sh" "07 bash scripts/07_axes.sh"
       "08 bash scripts/08_atlas.sh" "09 bash scripts/09_development.sh" "10 $PYT scripts/10_figures.py" "11 $PYT scripts/11_compare_with_previous.py"
       "12 $PYT scripts/12_presentation.py")
for s in "${STEPS[@]}"; do n=${s%% *}; cmd=${s#* }
  [ $((10#$n)) -lt $FROM ] && continue
  echo "== step $n: $cmd"; [ $DRY == 1 ] && continue
  eval "$cmd" || { echo "step $n failed"; exit 1; }
done
