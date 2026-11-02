#!/usr/bin/env bash
# Normal references compare with OTHER normal references, then correct query residuals.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
WORK_DIR="${WORK_DIR:-./checkpoint/normal_residual_pool_v1}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
REUSE_DIR="${REUSE_DIR:-auto}"
RUN_EXPORT="${RUN_EXPORT:-1}"
RUN_PREDICT="${RUN_PREDICT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PACKAGE="${RUN_PACKAGE:-1}"
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU_IDS' >&2; exit 1; }
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[${gpu}]:-}" ]] || { echo 'Duplicate GPU id' >&2; exit 1; }
  SEEN[${gpu}]=1
done
for flag in "${RUN_EXPORT}" "${RUN_PREDICT}" "${RUN_EVAL}" "${RUN_PACKAGE}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo 'RUN_* flags must be 0/1' >&2; exit 1; }
done
if [[ "${REUSE_DIR}" == auto ]]; then
  REUSE_DIR=""
  [[ ! -f ./checkpoint/residual_first_hr_text_v1/config.json ]] || REUSE_DIR=./checkpoint/residual_first_hr_text_v1
elif [[ "${REUSE_DIR}" == none ]]; then REUSE_DIR=""; fi
mkdir -p "${WORK_DIR}"
command -v flock >/dev/null || { echo 'Install util-linux (flock).' >&2; exit 1; }
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo 'Another process owns this WORK_DIR.' >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CPU_THREADS:-4}"
export MKL_NUM_THREADS="${CPU_THREADS:-4}"
echo 'Normal residual pool: each normal image matches ONLY the other normal images.'
echo 'D/P/S/N x global/high-resolution, no Flow or CLIP training, no hidden fallback.'
echo "Read-only V1 query cache reuse: ${REUSE_DIR:-disabled}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c \
  "import torch,numpy,PIL,scipy,sklearn,cv2; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; print('torch',torch.__version__)"
if [[ -z "${DATA_ROOTS_JSON:-}" ]]; then DATA_ROOTS_JSON='{}'; fi
DRIVER="${SCRIPT_DIR}/normal_pool_experiment.py"
COMMON=(--work_dir "${WORK_DIR}")
"${TRAIN_PYTHON}" "${DRIVER}" --stage prepare "${COMMON[@]}" --reuse_dir "${REUSE_DIR}" \
  --source "${SOURCE_DATASET:-visa}" --eval_datasets "${EVAL_DATASETS:-mvtec,btad,mpdd}" \
  --source_root "${SOURCE_ROOT:-}" --target_root "${TARGET_ROOT:-}" --data_roots "${DATA_ROOTS_JSON}" \
  --dino_repo_dir "${DINO_REPO_DIR:-./dinov3}" --dino_model_name "${DINO_MODEL_NAME:-dinov3_vitl16}" \
  --dino_weights "${DINO_WEIGHTS:-./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}" \
  --visual_layers "${VISUAL_LAYERS:-5,11,17,23}" --input_dim "${INPUT_DIM:-1024}" --patch_size "${PATCH_SIZE:-16}" \
  --image_size "${IMAGE_SIZE:-512}" --eval_size "${EVAL_SIZE:-512}" --feature_dim "${FEATURE_DIM:-256}" \
  --reference_count "${REFERENCE_COUNT:-4}" --tile_fraction "${TILE_FRACTION:-0.6}" --high_weight "${HIGH_WEIGHT:-0.25}" \
  --match_chunk "${MATCH_CHUNK:-128}" --bank_chunk "${BANK_CHUNK:-4096}" \
  --data_seed "${DATA_SEED:-42}" --num_shards "${#GPUS[@]}" --val_fraction "${VAL_FRACTION:-0.25}" \
  --limit_per_category "${LIMIT_PER_CATEGORY:-0}" --calibration_images "${CALIBRATION_IMAGES:-64}" \
  --pool_prior_weight "${POOL_PRIOR_WEIGHT:-0.5}" --floor_ratio "${POOL_FLOOR_RATIO:-0.01}" \
  --top_fraction "${TOP_FRACTION:-0.01}" --val_fpr "${VAL_FPR:-0.01}" --precision "${PRECISION:-bf16}"
PIDS=()
cleanup() { local pid; for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_shards() {
  local stage="$1" partition="$2" slot
  PIDS=()
  for slot in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[${slot}]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage "${stage}" "${COMMON[@]}" \
      --partition "${partition}" --shard_id "${slot}" &
    PIDS+=("$!")
  done
  local index
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[${index}]}"; then
      unset 'PIDS[index]'
      echo 'Worker failed: later stages will not start. Resume with unchanged settings.' >&2
      return 1
    fi
    unset 'PIDS[index]'
  done
  PIDS=()
}
run_partition() {
  local partition="$1"
  if [[ "${RUN_EXPORT}" == 1 ]]; then run_shards export "${partition}"; fi
  "${TRAIN_PYTHON}" "${DRIVER}" --stage seal "${COMMON[@]}" --partition "${partition}"
  if [[ "${partition}" == source ]]; then "${TRAIN_PYTHON}" "${DRIVER}" --stage fit "${COMMON[@]}"; fi
  if [[ "${RUN_PREDICT}" == 1 ]]; then run_shards predict "${partition}"; fi
  "${TRAIN_PYTHON}" "${DRIVER}" --stage lock_predictions "${COMMON[@]}" --partition "${partition}"
  if [[ "${RUN_EVAL}" == 1 ]]; then
    run_shards evaluate "${partition}"
    "${TRAIN_PYTHON}" "${DRIVER}" --stage report "${COMMON[@]}" --partition "${partition}"
  fi
}
run_partition source
run_partition eval
if [[ "${RUN_PACKAGE}" == 1 ]]; then "${TRAIN_PYTHON}" "${DRIVER}" --stage package "${COMMON[@]}"; fi
if [[ "${RUN_EVAL}" == 1 ]]; then
  echo "Completed. Results: ${WORK_DIR}/reports/eval/README.md"
else
  echo "Predictions locked. Evaluation was disabled (RUN_EVAL=0)."
fi
