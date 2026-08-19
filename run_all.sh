#!/usr/bin/env bash
# ------------------------------------------------------------------
#  MV-SRBF: train all, then infer all
# ------------------------------------------------------------------
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

echo -e "\n\033[1;33m##########  PHASE 1: TRAIN  ##########\033[0m"
bash "${PROJ_DIR}/train.sh"

echo -e "\n\033[1;33m##########  PHASE 2: INFER  ##########\033[0m"
bash "${PROJ_DIR}/infer.sh"

echo -e "\n\033[1;32mAll done.\033[0m"
