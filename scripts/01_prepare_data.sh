#!/bin/bash
# Step 01: inputs of the adult series and of the four young age groups (CPU, ~15 min)
source "$(dirname "$0")/common.sh"
$PYT -m ottr.data.series > logs/01_series.log 2>&1 || { echo "series FAILED"; exit 1; }
for x in "devP0 P0" "devP32 P32" "dev3M 3M" "dev7M 7M"; do set -- $x
  $PYT -m ottr.data.dev --ds $1 --stage $2 > logs/01_$1.log 2>&1 || { echo "$1 FAILED"; exit 1; }
done; echo "step 01 done"
