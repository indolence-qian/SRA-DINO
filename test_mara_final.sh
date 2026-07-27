#!/usr/bin/env bash
set -euo pipefail

# Evaluate final MARA-GRPO checkpoints with test_mara.py.
#
# Usage:
#   MARA_CKPT=./checkpoint/mara_visa_hfa3_xxx/ckpt bash test_mara_final.sh
#
# Optional overrides:
#   GPU_IDS=0,1 DATASETS="mvtec btad mpdd" bash test_mara_final.sh
#   EVAL_LATEST_ONLY=1 BATCH_SIZE=8 bash test_mara_final.sh
#   DISABLE_HARD_GAIN_GATE=1 bash test_mara_final.sh  # diagnose raw soft-gated refinement

TRICK_NAME="${TRICK_NAME:-mara_grpo_final}"
GPU_IDS="${GPU_IDS:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DATASETS="${DATASETS:-${DATASET:-mvtec btad mpdd}}"
MARA_CKPT="${MARA_CKPT:-}"
NORM_MODE="${NORM_MODE:-none}"
EVAL_LATEST_ONLY="${EVAL_LATEST_ONLY:-0}"
SAVE_VIS="${SAVE_VIS:-0}"
GAIN_SAFETY_MARGIN="${GAIN_SAFETY_MARGIN:-0.0}"
GAIN_ACCEPT_PROBABILITY="${GAIN_ACCEPT_PROBABILITY:-0.50}"
DISABLE_HARD_GAIN_GATE="${DISABLE_HARD_GAIN_GATE:-0}"

if [[ -z "${MARA_CKPT}" ]]; then
  echo "[ERROR] MARA_CKPT is required."
  echo "Example: MARA_CKPT=./checkpoint/mara_visa_hfa3_xxx/ckpt bash test_mara_final.sh"
  exit 1
fi

if [[ ! -e "${MARA_CKPT}" ]]; then
  echo "[ERROR] MARA_CKPT not found: ${MARA_CKPT}"
  exit 1
fi

TS="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="./logs/${TRICK_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/test_mara_final_${TS}.log"
RESULT_PATH="./TESTING_ALL/${TRICK_NAME}/${TS}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date +'%F %T')] Start MARA final evaluation"
echo "Log: ${LOG_FILE}"
echo "MARA checkpoint path: ${MARA_CKPT}"
echo "Result path: ${RESULT_PATH}"
echo "Datasets: ${DATASETS}"
echo "Parallel evaluation GPUs: ${GPU_IDS}"
echo "Hard gain gate: disabled=${DISABLE_HARD_GAIN_GATE}, safety_margin=${GAIN_SAFETY_MARGIN}, accept_probability=${GAIN_ACCEPT_PROBABILITY}"

LATEST_FLAG=()
if [[ "${EVAL_LATEST_ONLY}" == "1" ]]; then
  LATEST_FLAG=(--eval_latest_only)
fi

SAVE_VIS_FLAG=()
if [[ "${SAVE_VIS}" == "1" ]]; then
  SAVE_VIS_FLAG=(--save_vis)
fi

HARD_GAIN_GATE_FLAG=()
if [[ "${DISABLE_HARD_GAIN_GATE}" == "1" ]]; then
  HARD_GAIN_GATE_FLAG=(--disable_hard_gain_gate)
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
read -r -a DATASET_LIST <<< "${DATASETS}"

if (( ${#GPU_LIST[@]} == 0 )); then
  echo "[ERROR] GPU_IDS must contain at least one GPU id."
  exit 1
fi

if (( ${#DATASET_LIST[@]} == 0 )); then
  echo "[ERROR] DATASETS must contain at least one dataset."
  exit 1
fi

PIDS=()
for ((worker_idx = 0; worker_idx < ${#GPU_LIST[@]}; worker_idx++)); do
  gpu_id="${GPU_LIST[worker_idx]//[[:space:]]/}"
  (
    for ((dataset_idx = worker_idx; dataset_idx < ${#DATASET_LIST[@]}; dataset_idx += ${#GPU_LIST[@]})); do
      ds="${DATASET_LIST[dataset_idx]}"
      echo "===== GPU ${gpu_id}: Testing MARA on dataset ${ds} ====="
      CUDA_VISIBLE_DEVICES="${gpu_id}" python test_mara.py \
        --result_path "${RESULT_PATH}" \
        --weight_path "${MARA_CKPT}" \
        --device cuda:0 \
        --batch_size "${BATCH_SIZE}" \
        --dataset "${ds}" \
        --norm_mode "${NORM_MODE}" \
        --gain_safety_margin "${GAIN_SAFETY_MARGIN}" \
        --gain_accept_probability "${GAIN_ACCEPT_PROBABILITY}" \
        "${LATEST_FLAG[@]}" \
        "${SAVE_VIS_FLAG[@]}" \
        "${HARD_GAIN_GATE_FLAG[@]}"

      METRIC_FILE="${RESULT_PATH}/${ds}/metric_mara.txt"
      if [[ -f "${METRIC_FILE}" ]]; then
        echo "GPU ${gpu_id}: Metric file: ${METRIC_FILE}"
      else
        echo "[WARN] GPU ${gpu_id}: Metric file not found: ${METRIC_FILE}"
      fi
    done
  ) &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAILED=1
  fi
done

if [[ "${FAILED}" == "1" ]]; then
  echo "[ERROR] At least one evaluation worker failed."
  exit 1
fi

echo "===== MARA final evaluation finished ====="
echo "Results: ${RESULT_PATH}"
