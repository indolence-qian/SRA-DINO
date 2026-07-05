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
VISUAL_LAYERS="${VISUAL_LAYERS:-5,11,17,23}"
DEVICE="${DEVICE:-cuda:0}"

BASE_EPOCH="${BASE_EPOCH:-15}"
BASE_BS="${BASE_BS:-16}"
MARA_EPOCH="${MARA_EPOCH:-30}"
MARA_BS="${MARA_BS:-4}"
GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-4}"
MARA_STEPS="${MARA_STEPS:-3}"
MARA_DELTA_SCALE="${MARA_DELTA_SCALE:-0.35}"
MARA_GATE_MAX="${MARA_GATE_MAX:-0.50}"
MARA_GATE_INIT_BIAS="${MARA_GATE_INIT_BIAS:--3.5}"
GAIN_ACCEPT_THRESHOLD="${GAIN_ACCEPT_THRESHOLD:-0.0}"
GAIN_GATE_TEMPERATURE="${GAIN_GATE_TEMPERATURE:-0.1}"
GAIN_LOSS_CLIP="${GAIN_LOSS_CLIP:-1.0}"
DISABLE_GAIN_GATE="${DISABLE_GAIN_GATE:-0}"
BASE_ANCHOR_MARGIN="${BASE_ANCHOR_MARGIN:-0.0}"
NEGATIVE_ADVANTAGE_SCALE="${NEGATIVE_ADVANTAGE_SCALE:-1.0}"
ADVANTAGE_CLIP="${ADVANTAGE_CLIP:-5.0}"
W_BASE_CONSISTENCY="${W_BASE_CONSISTENCY:-0.05}"
W_GATE_SPARSE="${W_GATE_SPARSE:-0.01}"
W_GAIN_VALUE="${W_GAIN_VALUE:-0.05}"

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
echo "Dataset=${DATASET}, Device=${DEVICE}, HFA=${HFA_SETTING}, visual_layers=${VISUAL_LAYERS}"
echo "MARA gate: max=${MARA_GATE_MAX}, init_bias=${MARA_GATE_INIT_BIAS}, sparse_weight=${W_GATE_SPARSE}"
echo "Gain gate: disabled=${DISABLE_GAIN_GATE}, threshold=${GAIN_ACCEPT_THRESHOLD}, temperature=${GAIN_GATE_TEMPERATURE}, value_weight=${W_GAIN_VALUE}"

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
  GAIN_GATE_FLAG=()
  if [[ "${DISABLE_GAIN_GATE}" == "1" ]]; then
    GAIN_GATE_FLAG=(--disable_gain_gate)
  fi

  python train_mara.py \
    --result_path "${MARA_RESULT_PATH}" \
    --base_ckpt "${BASE_CKPT}" \
    --device "${DEVICE}" \
    --dataset "${DATASET}" \
    --batch_size "${MARA_BS}" \
    --epoch "${MARA_EPOCH}" \
    --visual_backbone dino \
    --visual_layers "${VISUAL_LAYERS}" \
    --hfa_setting "${HFA_SETTING}" \
    --mara_steps "${MARA_STEPS}" \
    --grpo_group_size "${GRPO_GROUP_SIZE}" \
    --mara_delta_scale "${MARA_DELTA_SCALE}" \
    --mara_gate_max "${MARA_GATE_MAX}" \
    --mara_gate_init_bias "${MARA_GATE_INIT_BIAS}" \
    --gain_accept_threshold "${GAIN_ACCEPT_THRESHOLD}" \
    --gain_gate_temperature "${GAIN_GATE_TEMPERATURE}" \
    --gain_loss_clip "${GAIN_LOSS_CLIP}" \
    --base_anchor_margin "${BASE_ANCHOR_MARGIN}" \
    --negative_advantage_scale "${NEGATIVE_ADVANTAGE_SCALE}" \
    --advantage_clip "${ADVANTAGE_CLIP}" \
    --w_base_consistency "${W_BASE_CONSISTENCY}" \
    --w_gate_sparse "${W_GATE_SPARSE}" \
    --w_gain_value "${W_GAIN_VALUE}" \
    "${GAIN_GATE_FLAG[@]}"
fi

echo "===== Pipeline finished ====="
echo "Base result: ${BASE_RESULT_PATH}"
echo "MARA result: ${MARA_RESULT_PATH}"
