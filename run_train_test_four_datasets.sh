#!/usr/bin/env bash
# Fresh current residual/Flow/HR/text network -> four fixed target datasets.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
WORK_DIR="${WORK_DIR:-./checkpoint/residual_hr_text_four_datasets_v1}"
GPU_IDS="${GPU_IDS:-0,1}"; GPU_IDS="${GPU_IDS//[[:space:]]/}"
EPOCHS="${EPOCHS:-15}"; TRAIN_UNTIL_EPOCH="${TRAIN_UNTIL_EPOCH:-${EPOCHS}}"
TRAIN_SEED="${TRAIN_SEED:-42}"
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ && "${TRAIN_SEED}" =~ ^[0-9]+$ ]] || { echo 'Invalid GPU_IDS / TRAIN_SEED' >&2; exit 1; }
[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ && "${TRAIN_UNTIL_EPOCH}" =~ ^[1-9][0-9]*$ && "${TRAIN_UNTIL_EPOCH}" -le "${EPOCHS}" ]] || { echo 'Require 1 <= TRAIN_UNTIL_EPOCH <= EPOCHS' >&2; exit 1; }
for flag in RUN_EXPORT RUN_TRAIN RUN_PREDICT RUN_EVAL; do
  value="${!flag:-1}"
  [[ "${value}" == 0 || "${value}" == 1 ]] || { echo "${flag} must be 0 or 1" >&2; exit 1; }
done
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[${gpu}]:-}" ]] || { echo 'Duplicate GPU' >&2; exit 1; }
  SEEN[${gpu}]=1
done
mkdir -p "${WORK_DIR}"
command -v flock >/dev/null || { echo 'flock is required (util-linux)' >&2; exit 1; }
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo 'Another run owns this WORK_DIR' >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run.log") 2>&1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPU_THREADS:-4}" MKL_NUM_THREADS="${CPU_THREADS:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_DATA_ROOT="${LD_DATA_ROOT:-/mnt/qfg/Tate_qfg/data/LD_AD_20261116}"
echo 'Train ONE fresh residual/Flow/HR/text model on VisA; test MVTec, BTAD, MPDD and LD.'
echo "Text=${TEXT_READOUT:-T2}, K=${REFERENCE_COUNT:-4}, epochs=${TRAIN_UNTIL_EPOCH}/${EPOCHS}, GPUs=${GPU_IDS}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c \
  "import torch,numpy,PIL,scipy,sklearn,cv2; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; print('torch',torch.__version__)"
"${TRAIN_PYTHON}" -c \
  'import sys; from pathlib import Path; from evaluate_ld import ensure_data; ensure_data(Path(sys.argv[1]),Path(sys.argv[2]))' \
  "${LD_DATA_ROOT}" "${DATA_ARCHIVE:-$(dirname -- "${LD_DATA_ROOT}")/LD_AD_20261116_bundle.tar.gz}"
DRIVER="${SCRIPT_DIR}/train_test_four_datasets.py"
COMMON=(--work_dir "${WORK_DIR}" --until_epoch "${TRAIN_UNTIL_EPOCH}" --train_seed "${TRAIN_SEED}")
if [[ -z "${DATA_ROOTS_JSON:-}" ]]; then DATA_ROOTS_JSON='{}'; fi
"${TRAIN_PYTHON}" "${DRIVER}" --stage prepare "${COMMON[@]}" \
  --source visa --eval_datasets mvtec,btad,mpdd,ld --source_root "${SOURCE_ROOT:-}" --data_roots "${DATA_ROOTS_JSON}" \
  --dino_repo_dir "${DINO_REPO_DIR:-./dinov3}" --dino_model_name "${DINO_MODEL_NAME:-dinov3_vitl16}" \
  --dino_weights "${DINO_WEIGHTS:-./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}" \
  --clip_weights "${CLIP_WEIGHTS:-./CLIP/ckpt/ViT-L-14-336px.pt}" \
  --visual_layers "${VISUAL_LAYERS:-5,11,17,23}" --input_dim "${INPUT_DIM:-1024}" --patch_size "${PATCH_SIZE:-16}" \
  --image_size "${IMAGE_SIZE:-512}" --eval_size "${EVAL_SIZE:-512}" --selection_size "${SELECTION_SIZE:-256}" \
  --feature_dim "${FEATURE_DIM:-256}" --reference_count "${REFERENCE_COUNT:-4}" \
  --tile_fraction "${TILE_FRACTION:-0.6}" --high_weight "${HIGH_WEIGHT:-0.25}" \
  --match_chunk "${MATCH_CHUNK:-128}" --bank_chunk "${BANK_CHUNK:-4096}" \
  --data_seed "${DATA_SEED:-42}" --num_shards "${#GPUS[@]}" --val_fraction "${VAL_FRACTION:-0.25}" \
  --limit_per_category "${LIMIT_PER_CATEGORY:-0}" --calibration_images "${CALIBRATION_IMAGES:-64}" \
  --density_epochs "${EPOCHS}" --text_epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE:-8}" --micro_batch "${MICRO_BATCH_SIZE:-1}" --workers "${LOADER_WORKERS:-2}" \
  --flow_blocks "${FLOW_BLOCKS:-8}" --hidden "${HEAD_HIDDEN:-256}" --tokens_per_view "${TOKENS_PER_VIEW:-256}" \
  --density_lr "${DENSITY_LR:-0.0001}" --text_lr "${TEXT_LR:-0.0001}" --precision "${PRECISION:-bf16}" \
  --val_fpr "${VAL_FPR:-0.01}" --text_strength "${TEXT_STRENGTH:-0.5}" --text_readout "${TEXT_READOUT:-T2}"
PIDS=()
cleanup() { local pid; for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_workers() {
  local index
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[${index}]}"; then
      unset 'PIDS[index]'; echo 'Worker failed; later stages will not run.' >&2; return 1
    fi
    unset 'PIDS[index]'
  done
  PIDS=()
}
run_shards() {
  local stage="$1" partition="${2:-eval}" slot
  PIDS=()
  for slot in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[${slot}]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage "${stage}" "${COMMON[@]}" \
      --partition "${partition}" --shard_id "${slot}" &
    PIDS+=("$!")
  done
  wait_workers
}
export_partition() {
  local partition="$1"
  if [[ "${RUN_EXPORT:-1}" == 1 ]]; then run_shards export "${partition}"; fi
  "${TRAIN_PYTHON}" "${DRIVER}" --stage seal "${COMMON[@]}" --partition "${partition}"
}
export_partition source
"${TRAIN_PYTHON}" "${DRIVER}" --stage fit_stats "${COMMON[@]}"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage text_bank "${COMMON[@]}"
if [[ "${RUN_TRAIN:-1}" == 1 ]]; then
  if [[ "${#GPUS[@]}" -ge 2 ]]; then
    PIDS=()
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage train "${COMMON[@]}" --kind density & PIDS+=("$!")
    CUDA_VISIBLE_DEVICES="${GPUS[1]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage train "${COMMON[@]}" --kind text & PIDS+=("$!")
    wait_workers
  else
    for kind in density text; do
      CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage train "${COMMON[@]}" --kind "${kind}"
    done
  fi
fi
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage select "${COMMON[@]}"
# All source training and selection finish before ANY target feature export.
if [[ "${RUN_PREDICT:-1}" == 1 || "${RUN_EVAL:-1}" == 1 ]]; then
  export_partition eval
  if [[ "${RUN_PREDICT:-1}" == 1 ]]; then run_shards predict; fi
  "${TRAIN_PYTHON}" "${DRIVER}" --stage lock_predictions "${COMMON[@]}"
  if [[ "${RUN_EVAL:-1}" == 1 ]]; then
    run_shards evaluate
    "${TRAIN_PYTHON}" "${DRIVER}" --stage report "${COMMON[@]}"
  fi
fi
echo "Four-dataset results: ${WORK_DIR}/reports/FOUR_DATASETS.md"
echo "Download report: ${WORK_DIR}/four_datasets_report.zip"
