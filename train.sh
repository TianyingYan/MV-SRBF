#!/usr/bin/env bash
# ------------------------------------------------------------------
#  MV-SRBF: train every dataset experiment sequentially
# ------------------------------------------------------------------
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

EXPERIMENTS=(
    rgbn300
    rgbnt100
    msvr310
    wmveid863
)

log()  { echo -e "\n\033[1;36m===== $1 =====\033[0m"; }
fail() { echo -e "\033[1;31m[FAIL] $1\033[0m"; FAILED+=("$1"); }

FAILED=()

for exp in "${EXPERIMENTS[@]}"; do
    log "TRAIN  ${exp}"
    if python tools/train.py --config "configs/experiments/${exp}.yaml"; then
        echo -e "\033[1;32m[OK] ${exp}\033[0m"
    else
        fail "${exp}"
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All training completed successfully."
else
    echo "Failed: ${FAILED[*]}"
    exit 1
fi
