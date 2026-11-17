#!/usr/bin/env bash
# Original immutable training/inference -> all-sample as-provided evaluation.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
WORK_DIR="${WORK_DIR:-./checkpoint/text_comparison_v1}"
TRAIN_UNTIL_EPOCH="${TRAIN_UNTIL_EPOCH:-5}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"
# Keep original code hashes and exact learned models. Only bypass its invalid
# nonempty-mask check by doing evaluation through the corrected v3 entry point.
env WORK_DIR="${WORK_DIR}" TRAIN_UNTIL_EPOCH="${TRAIN_UNTIL_EPOCH}" TRAIN_SEEDS="${TRAIN_SEEDS}" \
  RUN_EVAL=0 RUN_PACKAGE=0 UPDATE_REPO_REPORT=0 bash "${SCRIPT_DIR}/run_exp_text_alignment_ablation.sh"
env WORK_DIR="${WORK_DIR}" TRAIN_UNTIL_EPOCH="${TRAIN_UNTIL_EPOCH}" TRAIN_SEEDS="${TRAIN_SEEDS}" \
  bash "${SCRIPT_DIR}/run_repair_text_evaluation.sh"
