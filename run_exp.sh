#!/usr/bin/env bash
set -euo pipefail

# DINO single-tower experiment:
#   1) Train the language-free DINO visual-prototype detector with DDP.
#   2) Freeze it and train MARA-GRPO with compact DINO feature evidence.
#   3) Evaluate the latest MARA checkpoint on unseen datasets in parallel.
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

MARA_EPOCH="${MARA_EPOCH:-30}"
MARA_BS="${MARA_BS:-2}"
TEST_BS="${TEST_BS:-16}"
TEST_EVAL_LATEST_ONLY="${TEST_EVAL_LATEST_ONLY:-1}"
TRICK_NAME="${TRICK_NAME:-dino_single_mara}"
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

if [[ "${RUN_MARA}" == "1" || "${RUN_TEST}" == "1" ]]; then
  echo "===== Stage 2/3: MARA training and cross-dataset evaluation ====="
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

echo "===== DINO single-tower experiment finished ====="
echo "Base checkpoint: ${BASE_CKPT}"
