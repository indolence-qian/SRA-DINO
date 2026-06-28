#!/usr/bin/env bash
set -euo pipefail

# Two-stage SRA-DINO training:
#   1) Train the base SRA-DINOv3 detector.
#   2) Freeze the base detector and train the MARA-GRPO refinement agent.
#
# Common overrides:
#   DEVICE=cuda:1 bash train_mara_visa.sh
#   BASE_EPOCH=30 MARA_EPOCH=50 bash train_mara_visa.sh
#   BASE_CKPT=./checkpoint/base_visa_hfa3_xxx/ckpt/14.pth RUN_BASE=0 bash train_mara_visa.sh

TRICK_NAME="${TRICK_NAME:-mara_grpo}"
DATASET="${DATASET:-visa}"
HFA_SETTING="${HFA_SETTING:-hfa3}"
DEVICE="${DEVICE:-cuda:0}"

BASE_EPOCH="${BASE_EPOCH:-15}"
BASE_BS="${BASE_BS:-16}"
MARA_EPOCH="${MARA_EPOCH:-30}"
MARA_BS="${MARA_BS:-4}"
GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-4}"
MARA_STEPS="${MARA_STEPS:-3}"

RUN_BASE="${RUN_BASE:-1}"
RUN_MARA="${RUN_MARA:-1}"
BASE_CKPT="${BASE_CKPT:-}"

TS="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="./logs/${TRICK_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_mara_${DATASET}_${TS}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date +'%F %T')] Start MARA training pipeline"
echo "Log: ${LOG_FILE}"
echo "Dataset=${DATASET}, Device=${DEVICE}, HFA=${HFA_SETTING}"

BASE_RESULT_PATH="./checkpoint/base_${DATASET}_${HFA_SETTING}_${TS}"
MARA_RESULT_PATH="./checkpoint/mara_${DATASET}_${HFA_SETTING}_${TS}"

if [[ "${RUN_BASE}" == "1" ]]; then
  echo "===== Stage 1: Train base SRA-DINOv3 detector ====="
  python train.py \
    --result_path "${BASE_RESULT_PATH}" \
    --device "${DEVICE}" \
    --dataset "${DATASET}" \
    --epoch "${BASE_EPOCH}" \
    --batch_size "${BASE_BS}" \
    --hfa_setting "${HFA_SETTING}"

  BASE_CKPT="$(find "${BASE_RESULT_PATH}/ckpt" -maxdepth 1 -name '*.pth' -printf '%f\n' \
    | sort -V \
    | tail -n 1)"
  BASE_CKPT="${BASE_RESULT_PATH}/ckpt/${BASE_CKPT}"
fi

if [[ -z "${BASE_CKPT}" ]]; then
  echo "[ERROR] BASE_CKPT is empty. Either run base training or pass BASE_CKPT=path/to/base.pth."
  exit 1
fi

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "[ERROR] BASE_CKPT not found: ${BASE_CKPT}"
  exit 1
fi

echo "Base checkpoint: ${BASE_CKPT}"

if [[ "${RUN_MARA}" == "1" ]]; then
  echo "===== Stage 2: Train MARA-GRPO refinement agent ====="
  python train_mara.py \
    --result_path "${MARA_RESULT_PATH}" \
    --base_ckpt "${BASE_CKPT}" \
    --device "${DEVICE}" \
    --dataset "${DATASET}" \
    --batch_size "${MARA_BS}" \
    --epoch "${MARA_EPOCH}" \
    --visual_backbone dino \
    --visual_layers 5,11,17,23 \
    --mara_steps "${MARA_STEPS}" \
    --grpo_group_size "${GRPO_GROUP_SIZE}"
fi

echo "===== Pipeline finished ====="
echo "Base result: ${BASE_RESULT_PATH}"
echo "MARA result: ${MARA_RESULT_PATH}"
