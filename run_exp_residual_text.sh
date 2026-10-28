#!/usr/bin/env bash
# New residual-first pipeline. Never reads or rewrites legacy Base/H/T/R caches.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
WORK_DIR="${WORK_DIR:-./checkpoint/residual_first_hr_text_v1}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"
TRAIN_SEEDS="${TRAIN_SEEDS//[[:space:]]/}"
EPOCHS="${EPOCHS:-15}"
TRAIN_UNTIL_EPOCH="${TRAIN_UNTIL_EPOCH:-5}"
RUN_EXPORT="${RUN_EXPORT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_PREDICT="${RUN_PREDICT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ && "${TRAIN_SEEDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo 'GPU_IDS / TRAIN_SEEDS must be comma-separated distinct nonnegative integers.' >&2; exit 1;
}
for flag in "${RUN_EXPORT}" "${RUN_TRAIN}" "${RUN_PREDICT}" "${RUN_EVAL}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo 'RUN_* flags must be 0/1' >&2; exit 1; }
done
[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ && "${TRAIN_UNTIL_EPOCH}" =~ ^[1-9][0-9]*$ && "${TRAIN_UNTIL_EPOCH}" -le "${EPOCHS}" ]] || {
  echo 'Require 1 <= TRAIN_UNTIL_EPOCH <= EPOCHS. Keep EPOCHS fixed on resume.' >&2; exit 1;
}
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
IFS=',' read -r -a SEEDS <<< "${TRAIN_SEEDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[g${gpu}]:-}" ]] || { echo 'Duplicate GPU id' >&2; exit 1; }
  SEEN[g${gpu}]=1
done
for seed in "${SEEDS[@]}"; do
  [[ -z "${SEEN[s${seed}]:-}" ]] || { echo 'Duplicate train seed' >&2; exit 1; }
  SEEN[s${seed}]=1
done
mkdir -p "${WORK_DIR}"
command -v flock >/dev/null || { echo 'Install util-linux (flock).' >&2; exit 1; }
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo 'Another pipeline owns this WORK_DIR.' >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
export PYTHONUNBUFFERED=1
echo 'Residual-first: normal references -> frozen pretrained DINO -> signed residuals -> Flow / text delta.'
echo 'NO Base checkpoint, HFA, old Prompt Adapter, VLM, RL or target fine-tuning.'
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c \
  "import torch,numpy,PIL,scipy,sklearn,cv2; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; print('torch',torch.__version__)"
if [[ -z "${DATA_ROOTS_JSON:-}" ]]; then DATA_ROOTS_JSON='{}'; fi
DRIVER="${SCRIPT_DIR}/residual_text_experiment.py"
COMMON=(--work_dir "${WORK_DIR}" --until_epoch "${TRAIN_UNTIL_EPOCH}")
"${TRAIN_PYTHON}" "${DRIVER}" --stage prepare "${COMMON[@]}" \
  --source "${SOURCE_DATASET:-visa}" --eval_datasets "${EVAL_DATASETS:-mvtec,btad,mpdd}" \
  --source_root "${SOURCE_ROOT:-}" --target_root "${TARGET_ROOT:-}" --data_roots "${DATA_ROOTS_JSON}" \
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
  --val_fpr "${VAL_FPR:-0.01}" --text_strengths "${TEXT_STRENGTHS:-0,0.05,0.1,0.25,0.5}" \
  --semantic_annotations "${SEMANTIC_ANNOTATIONS:-}"
# Validate the explicit CLIP checkpoint and cache the frozen text directions
# before spending hours on DINO export. No CLIP image forward is performed.
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage text_bank "${COMMON[@]}"
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_workers() {
  local index
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[${index}]}"; then
      unset 'PIDS[index]'
      echo 'Worker failed: no later stage will start. Resume with unchanged settings.' >&2
      return 1
    fi
    unset 'PIDS[index]'
  done
  PIDS=()
}
run_shards() {
  local stage="$1" partition="$2" seed="${3:-42}" slot
  PIDS=()
  for slot in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[${slot}]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage "${stage}" "${COMMON[@]}" \
      --partition "${partition}" --shard_id "${slot}" --train_seed "${seed}" &
    PIDS+=("$!")
  done
  wait_workers
}
export_partition() {
  local partition="$1"
  if [[ "${RUN_EXPORT}" == 1 ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage export "${COMMON[@]}" \
      --partition "${partition}" --shard_id 0 --probe
    run_shards export "${partition}"
  fi
  "${TRAIN_PYTHON}" "${DRIVER}" --stage seal "${COMMON[@]}" --partition "${partition}"
}
train_variants() {
  local stage="$1" seed="$2" offset slot index
  local variants=(G HR)
  for ((offset=0; offset<2; offset+=${#GPUS[@]})); do
    PIDS=()
    for slot in "${!GPUS[@]}"; do
      index=$((offset+slot))
      [[ "${index}" -lt 2 ]] || break
      CUDA_VISIBLE_DEVICES="${GPUS[${slot}]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage "${stage}" "${COMMON[@]}" \
        --variant "${variants[${index}]}" --train_seed "${seed}" &
      PIDS+=("$!")
    done
    wait_workers
  done
}
export_partition source
"${TRAIN_PYTHON}" "${DRIVER}" --stage fit_stats "${COMMON[@]}"
for seed in "${SEEDS[@]}"; do
  if [[ "${RUN_TRAIN}" == 1 ]]; then
    train_variants train_density "${seed}"
    train_variants train_text "${seed}"
  fi
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage select "${COMMON[@]}" --train_seed "${seed}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage audit "${COMMON[@]}" --train_seed "${seed}"
done
if [[ "${RUN_PREDICT}" == 1 || "${RUN_EVAL}" == 1 ]]; then
  export_partition eval
  for seed in "${SEEDS[@]}"; do
    if [[ "${RUN_PREDICT}" == 1 ]]; then run_shards predict eval "${seed}"; fi
    "${TRAIN_PYTHON}" "${DRIVER}" --stage lock_predictions "${COMMON[@]}" --train_seed "${seed}"
    if [[ "${RUN_EVAL}" == 1 ]]; then run_shards evaluate eval "${seed}"; fi
  done
fi
if [[ "${RUN_EVAL}" == 1 ]]; then
  "${TRAIN_PYTHON}" "${DRIVER}" --stage report "${COMMON[@]}" --train_seeds "${TRAIN_SEEDS}"
fi
echo "Completed through${TRAIN_UNTIL_EPOCH}. Reports: ${WORK_DIR}/reports/through${TRAIN_UNTIL_EPOCH}/"
