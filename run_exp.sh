#!/usr/bin/env bash
set -euo pipefail

# DINO + offline Qwen3-VL teacher experiment:
#   1) Train the language-free DINO visual-prototype detector with DDP.
#   2) Run one independent Qwen3-VL-8B-FP8 teacher per GPU and cache decisions.
#   3) Distill the cached decisions into a compact DINO semantic head with DDP.
#   4) Freeze stage one and train/evaluate MARA-GRPO with semantic evidence.
#
# Full run:
#   GPU_IDS=0,1 NPROC_PER_NODE=2 bash run_exp.sh
# Resume from a trained single-tower base:
#   RUN_BASE=0 BASE_CKPT=./checkpoint/dino_single_visa_xxx/ckpt/single_epoch_14.pth bash run_exp.sh

SOURCE_DATASET="${SOURCE_DATASET:-visa}"
TRAIN_SPLIT="${TRAIN_SPLIT:-test}"
TEST_DATASETS="${TEST_DATASETS:-mvtec btad mpdd}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DEVICE="${DEVICE:-cuda:0}"

RUN_BASE="${RUN_BASE:-1}"
RUN_VLM_CACHE="${RUN_VLM_CACHE:-1}"
RUN_VLM_DISTILL="${RUN_VLM_DISTILL:-1}"
RUN_MARA="${RUN_MARA:-1}"
RUN_TEST="${RUN_TEST:-1}"
BASE_CKPT="${BASE_CKPT:-}"

BASE_EPOCH="${BASE_EPOCH:-15}"
BASE_BS="${BASE_BS:-4}"
BASE_LR="${BASE_LR:-1e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
VISUAL_LAYERS="${VISUAL_LAYERS:-5,11,17,23}"
EMBED_DIM="${EMBED_DIM:-256}"
NORMAL_PROTOTYPES="${NORMAL_PROTOTYPES:-4}"
ANOMALY_PROTOTYPES="${ANOMALY_PROTOTYPES:-8}"
EVIDENCE_CHANNELS="${EVIDENCE_CHANNELS:-8}"
SINGLE_HFA_SETTING="${SINGLE_HFA_SETTING:-none}"

VLM_MODEL_ID="${VLM_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
VLM_PYTHON="${VLM_PYTHON:-python}"
VLM_GPU_MEMORY="${VLM_GPU_MEMORY:-0.70}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-4096}"
VLM_MAX_TOKENS="${VLM_MAX_TOKENS:-192}"
VLM_NUM_ROIS="${VLM_NUM_ROIS:-3}"
VLM_IMAGE_SIZE="${VLM_IMAGE_SIZE:-512}"
VLM_ROI_FRACTION="${VLM_ROI_FRACTION:-0.25}"
VLM_MAX_INVALID_RATIO="${VLM_MAX_INVALID_RATIO:-0.05}"
VLM_CACHE_PATH="${VLM_CACHE_PATH:-}"
VLM_WORK_DIR="${VLM_WORK_DIR:-}"
VLM_DISTILL_EPOCH="${VLM_DISTILL_EPOCH:-5}"
VLM_DISTILL_BS="${VLM_DISTILL_BS:-4}"
VLM_DISTILL_LR="${VLM_DISTILL_LR:-3e-4}"
VLM_HIDDEN_DIM="${VLM_HIDDEN_DIM:-64}"
VLM_MAP_SIZE="${VLM_MAP_SIZE:-32}"

MARA_EPOCH="${MARA_EPOCH:-30}"
MARA_BS="${MARA_BS:-2}"
TEST_BS="${TEST_BS:-16}"
TEST_EVAL_LATEST_ONLY="${TEST_EVAL_LATEST_ONLY:-1}"
TRICK_NAME="${TRICK_NAME:-dino_single_qwen3vl_mara}"
TEST_TRICK_NAME="${TEST_TRICK_NAME:-${TRICK_NAME}_final}"

DINO_REPO_DIR="${DINO_REPO_DIR:-./dinov3}"
DINO_MODEL_NAME="${DINO_MODEL_NAME:-dinov3_vitl16}"
DINO_WEIGHTS="${DINO_WEIGHTS:-./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if (( ${#GPU_LIST[@]} != NPROC_PER_NODE )); then
  echo "[ERROR] GPU_IDS contains ${#GPU_LIST[@]} GPUs but NPROC_PER_NODE=${NPROC_PER_NODE}."
  exit 1
fi

if [[ "${RUN_TEST}" == "1" && "${RUN_MARA}" != "1" ]]; then
  echo "[ERROR] RUN_TEST=1 requires RUN_MARA=1 in this integrated script."
  exit 1
fi

if [[ ! -f "${DINO_WEIGHTS}" ]]; then
  echo "[ERROR] DINOv3 weights not found: ${DINO_WEIGHTS}"
  exit 1
fi

if command -v torchrun >/dev/null 2>&1; then
  LAUNCHER=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}")
else
  LAUNCHER=(python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}")
fi

TS="$(date +'%Y%m%d_%H%M%S')"
BASE_RESULT_PATH="./checkpoint/dino_single_${SOURCE_DATASET}_${TS}"
LOG_DIR="./logs/${TRICK_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_exp_${SOURCE_DATASET}_${TS}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date +'%F %T')] Start DINO single-tower experiment"
echo "Log: ${LOG_FILE}"
echo "Source=${SOURCE_DATASET}/${TRAIN_SPLIT}, targets=${TEST_DATASETS}"
echo "GPUs=${GPU_IDS}, processes=${NPROC_PER_NODE}"
echo "Prototypes normal/anomaly=${NORMAL_PROTOTYPES}/${ANOMALY_PROTOTYPES}, embed=${EMBED_DIM}"
echo "DINO feature evidence=${EVIDENCE_CHANNELS} channels/layer"
echo "VLM teacher=${VLM_MODEL_ID}, offline workers=${#GPU_LIST[@]}"

if [[ "${RUN_BASE}" == "1" || "${RUN_VLM_DISTILL}" == "1" || "${RUN_MARA}" == "1" || "${RUN_TEST}" == "1" ]]; then
  if ! python -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= ${NPROC_PER_NODE}; print('Training CUDA:', torch.__version__, torch.version.cuda, torch.cuda.device_count())"; then
    echo "[ERROR] The active training environment cannot access ${NPROC_PER_NODE} CUDA GPUs."
    echo "[ERROR] Do not install vLLM into the training environment; restore its CUDA-compatible PyTorch first."
    exit 1
  fi
fi

if [[ "${RUN_BASE}" == "1" ]]; then
  echo "===== Stage 1: DINO single-tower visual-prototype training ====="
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${LAUNCHER[@]}" train_dino_single.py \
    --result_path "${BASE_RESULT_PATH}" \
    --device "${DEVICE}" \
    --dataset "${SOURCE_DATASET}" \
    --train_split "${TRAIN_SPLIT}" \
    --image_size "${IMAGE_SIZE}" \
    --batch_size "${BASE_BS}" \
    --epoch "${BASE_EPOCH}" \
    --lr "${BASE_LR}" \
    --visual_layers "${VISUAL_LAYERS}" \
    --embed_dim "${EMBED_DIM}" \
    --normal_prototypes "${NORMAL_PROTOTYPES}" \
    --anomaly_prototypes "${ANOMALY_PROTOTYPES}" \
    --evidence_channels "${EVIDENCE_CHANNELS}" \
    --hfa_setting "${SINGLE_HFA_SETTING}" \
    --dino_repo_dir "${DINO_REPO_DIR}" \
    --dino_model_name "${DINO_MODEL_NAME}" \
    --dino_weights "${DINO_WEIGHTS}"

  BASE_NAME="$(find "${BASE_RESULT_PATH}/ckpt" -maxdepth 1 -name 'single_epoch_*.pth' -printf '%f\n' \
    | sort -V \
    | tail -n 1)"
  BASE_CKPT="${BASE_RESULT_PATH}/ckpt/${BASE_NAME}"
fi

if [[ -z "${BASE_CKPT}" || ! -f "${BASE_CKPT}" ]]; then
  echo "[ERROR] A valid DINO single-tower BASE_CKPT is required: ${BASE_CKPT:-<empty>}"
  exit 1
fi
echo "Single-tower base checkpoint: ${BASE_CKPT}"

if [[ -z "${VLM_WORK_DIR}" ]]; then
  VLM_WORK_DIR="./checkpoint/vlm_teacher_${SOURCE_DATASET}_${TS}"
fi
if [[ -z "${VLM_CACHE_PATH}" ]]; then
  VLM_CACHE_PATH="${VLM_WORK_DIR}/qwen3_vl_fp8_decisions.jsonl"
fi

if [[ "${RUN_VLM_CACHE}" == "1" ]]; then
  echo "===== Stage 2: dual-GPU Qwen3-VL-8B-FP8 decision cache ====="
  mkdir -p "${VLM_WORK_DIR}"
  if ! "${VLM_PYTHON}" -c "import qwen_vl_utils, torch, transformers, vllm; assert torch.cuda.is_available(); print('VLM CUDA:', torch.__version__, torch.version.cuda, vllm.__version__)"; then
    echo "[ERROR] VLM_PYTHON does not provide a working CUDA/vLLM environment."
    echo "[ERROR] Create the separate CUDA 12.8 environment documented in README.md."
    exit 1
  fi
  VLM_PIDS=()
  VLM_SHARDS="${#GPU_LIST[@]}"
  for shard_id in "${!GPU_LIST[@]}"; do
    gpu_id="${GPU_LIST[${shard_id}]}"
    echo "Starting VLM shard ${shard_id}/${VLM_SHARDS} on physical GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${VLM_PYTHON}" build_vlm_decision_cache.py \
      --base_ckpt "${BASE_CKPT}" \
      --output_path "${VLM_CACHE_PATH}" \
      --shard_id "${shard_id}" \
      --num_shards "${VLM_SHARDS}" \
      --dataset "${SOURCE_DATASET}" \
      --train_split "${TRAIN_SPLIT}" \
      --image_size "${IMAGE_SIZE}" \
      --device cuda:0 \
      --model_id "${VLM_MODEL_ID}" \
      --gpu_memory_utilization "${VLM_GPU_MEMORY}" \
      --max_model_len "${VLM_MAX_MODEL_LEN}" \
      --max_tokens "${VLM_MAX_TOKENS}" \
      --num_rois "${VLM_NUM_ROIS}" \
      --teacher_image_size "${VLM_IMAGE_SIZE}" \
      --roi_fraction "${VLM_ROI_FRACTION}" \
      --max_invalid_ratio "${VLM_MAX_INVALID_RATIO}" \
      --dino_repo_dir "${DINO_REPO_DIR}" \
      --dino_model_name "${DINO_MODEL_NAME}" \
      --dino_weights "${DINO_WEIGHTS}" &
    VLM_PIDS+=("$!")
  done
  VLM_FAILED=0
  for pid in "${VLM_PIDS[@]}"; do
    if ! wait "${pid}"; then
      VLM_FAILED=1
    fi
  done
  if [[ "${VLM_FAILED}" != "0" ]]; then
    echo "[ERROR] At least one VLM cache worker failed. Shards are resumable; rerun the same command."
    exit 1
  fi
  "${VLM_PYTHON}" build_vlm_decision_cache.py \
    --output_path "${VLM_CACHE_PATH}" \
    --num_shards "${VLM_SHARDS}" \
    --merge_shards
fi

if [[ "${RUN_VLM_DISTILL}" == "1" ]]; then
  if [[ ! -f "${VLM_CACHE_PATH}" ]]; then
    echo "[ERROR] VLM cache not found: ${VLM_CACHE_PATH}"
    exit 1
  fi
  echo "===== Stage 3: dual-GPU semantic decision-head distillation ====="
  VLM_DISTILL_RESULT="${VLM_WORK_DIR}/distilled"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${LAUNCHER[@]}" train_vlm_distiller.py \
    --base_ckpt "${BASE_CKPT}" \
    --cache_path "${VLM_CACHE_PATH}" \
    --result_path "${VLM_DISTILL_RESULT}" \
    --device "${DEVICE}" \
    --dataset "${SOURCE_DATASET}" \
    --train_split "${TRAIN_SPLIT}" \
    --image_size "${IMAGE_SIZE}" \
    --map_size "${VLM_MAP_SIZE}" \
    --batch_size "${VLM_DISTILL_BS}" \
    --epoch "${VLM_DISTILL_EPOCH}" \
    --lr "${VLM_DISTILL_LR}" \
    --hidden_dim "${VLM_HIDDEN_DIM}" \
    --teacher_model "${VLM_MODEL_ID}" \
    --dino_repo_dir "${DINO_REPO_DIR}" \
    --dino_model_name "${DINO_MODEL_NAME}" \
    --dino_weights "${DINO_WEIGHTS}"

  DISTILLED_NAME="$(find "${VLM_DISTILL_RESULT}/ckpt" -maxdepth 1 \
    -name 'vlm_distilled_epoch_*.pth' -printf '%f\n' | sort -V | tail -n 1)"
  BASE_CKPT="${VLM_DISTILL_RESULT}/ckpt/${DISTILLED_NAME}"
  if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "[ERROR] Distilled semantic checkpoint was not created."
    exit 1
  fi
  echo "Distilled semantic checkpoint: ${BASE_CKPT}"
fi

if [[ "${RUN_MARA}" == "1" || "${RUN_TEST}" == "1" ]]; then
  echo "===== Stage 4/5: MARA training and cross-dataset evaluation ====="
  RUN_BASE=0 \
  RUN_MARA="${RUN_MARA}" \
  RUN_TEST="${RUN_TEST}" \
  BASE_CKPT="${BASE_CKPT}" \
  DATASET="${SOURCE_DATASET}" \
  TRAIN_SPLIT="${TRAIN_SPLIT}" \
  DEVICE="${DEVICE}" \
  GPU_IDS="${GPU_IDS}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  MARA_EPOCH="${MARA_EPOCH}" \
  MARA_BS="${MARA_BS}" \
  TEST_DATASETS="${TEST_DATASETS}" \
  TEST_BS="${TEST_BS}" \
  TEST_EVAL_LATEST_ONLY="${TEST_EVAL_LATEST_ONLY}" \
  TRICK_NAME="${TRICK_NAME}" \
  TEST_TRICK_NAME="${TEST_TRICK_NAME}" \
  HFA_SETTING="${SINGLE_HFA_SETTING}" \
  VISUAL_LAYERS="${VISUAL_LAYERS}" \
  DINO_REPO_DIR="${DINO_REPO_DIR}" \
  DINO_MODEL_NAME="${DINO_MODEL_NAME}" \
  DINO_WEIGHTS="${DINO_WEIGHTS}" \
  bash train_mara_visa.sh
fi

echo "===== DINO + Qwen3-VL teacher experiment finished ====="
echo "Stage-one checkpoint used by MARA: ${BASE_CKPT}"
echo "VLM decision cache: ${VLM_CACHE_PATH}"
