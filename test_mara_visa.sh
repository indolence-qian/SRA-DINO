#!/usr/bin/env bash
set -euo pipefail

# Evaluate base SRA-DINOv3 checkpoints with the existing test2.py evaluator.
#
# Usage:
#   WEIGHT_PATH=./checkpoint/base_visa_hfa3_xxx/ckpt bash test_mara_visa.sh
#
# Optional overrides:
#   DEVICE=cuda:1 DATASETS="mvtec btad mpdd" bash test_mara_visa.sh
#   MAX_EPOCH=15 BATCH_SIZE=32 SAVE_VIS=1 bash test_mara_visa.sh
#
# Note:
#   This script evaluates the base detector checkpoints used by MARA.
#   MARA-specific inference needs a dedicated evaluator that loads mara_agent.

TRICK_NAME="${TRICK_NAME:-mara_grpo}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCH="${MAX_EPOCH:-100}"
HFA_SETTING="${HFA_SETTING:-hfa3}"
NORM_MODE="${NORM_MODE:-none}"
SAVE_VIS="${SAVE_VIS:-0}"
DATASETS="${DATASETS:-mvtec btad mpdd}"
WEIGHT_PATH="${WEIGHT_PATH:-}"

if [[ -z "${WEIGHT_PATH}" ]]; then
  echo "[ERROR] WEIGHT_PATH is required."
  echo "Example: WEIGHT_PATH=./checkpoint/base_visa_hfa3_xxx/ckpt bash test_mara_visa.sh"
  exit 1
fi

if [[ ! -d "${WEIGHT_PATH}" ]]; then
  echo "[ERROR] WEIGHT_PATH must be a checkpoint directory: ${WEIGHT_PATH}"
  exit 1
fi

TS="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="./logs/${TRICK_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/test_base_${TS}.log"
RESULT_PATH="./TESTING_ALL/${TRICK_NAME}/${TS}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date +'%F %T')] Start base checkpoint evaluation"
echo "Log: ${LOG_FILE}"
echo "Weight path: ${WEIGHT_PATH}"
echo "Result path: ${RESULT_PATH}"
echo "Datasets: ${DATASETS}"

SAVE_VIS_FLAG=()
if [[ "${SAVE_VIS}" == "1" ]]; then
  SAVE_VIS_FLAG=(--save_vis)
fi

for ds in ${DATASETS}; do
  echo "===== Testing dataset: ${ds} ====="
  python test2.py \
    --result_path "${RESULT_PATH}" \
    --weight_path "${WEIGHT_PATH}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --dataset "${ds}" \
    --hfa_setting "${HFA_SETTING}" \
    --auto_hfa_from_ckpt \
    --norm_mode "${NORM_MODE}" \
    --max_epoch "${MAX_EPOCH}" \
    "${SAVE_VIS_FLAG[@]}"

  METRIC_FILE="${RESULT_PATH}/${ds}/metric.txt"
  if [[ -f "${METRIC_FILE}" ]]; then
    echo "Metric file: ${METRIC_FILE}"
  else
    echo "[WARN] Metric file not found: ${METRIC_FILE}"
  fi
done

echo "===== Evaluation finished ====="
echo "Results: ${RESULT_PATH}"
