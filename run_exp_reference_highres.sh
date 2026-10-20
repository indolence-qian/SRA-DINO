#!/usr/bin/env bash
set -euo pipefail
# Frozen Base -> label-free global/local/reference cache -> four matched heads -> targets.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
WORK_DIR="${WORK_DIR:-./checkpoint/reference_highres_v1}"
BASE_CKPT="${BASE_CKPT:-}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
RUN_EXPORT="${RUN_EXPORT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
[[ -e "${BASE_CKPT}" ]] || { echo 'Set BASE_CKPT to an existing frozen CLIP+DINO Base file/directory.' >&2; exit 1; }
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'GPU_IDS must be distinct comma-separated integer IDs.' >&2; exit 1; }
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[${gpu}]:-}" ]] || { echo 'Duplicate GPU id' >&2; exit 1; }
  SEEN[${gpu}]=1
done
for flag in "${RUN_EXPORT}" "${RUN_TRAIN}" "${RUN_EVAL}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo 'Stage flags must be 0/1' >&2; exit 1; }
done
mkdir -p "${WORK_DIR}"
command -v flock >/dev/null || { echo 'Install util-linux (flock).' >&2; exit 1; }
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo 'Another pipeline is using this WORK_DIR.' >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
export PYTHONUNBUFFERED=1
echo 'A_global / B_highres / C_reference / D_highres_reference; frozen CLIP+DINO, NO VLM or MARA'
echo "Source=${SOURCE_DATASET:-visa}, targets=${EVAL_DATASETS:-mvtec,btad,mpdd}; C/D use known-normal target TRAIN support"
echo "GPUs=${GPU_IDS}; effective batch=${HEAD_BATCH_SIZE:-8}; micro-batch=${MICRO_BATCH_SIZE:-2}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c "import torch,numpy,PIL,cv2,scipy,sklearn; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; print('torch:',torch.__version__)"
if [[ -z "${DATA_ROOTS_JSON:-}" ]]; then DATA_ROOTS_JSON='{}'; fi
"${TRAIN_PYTHON}" reference_highres_experiment.py --stage prepare --work_dir "${WORK_DIR}" \
  --base_ckpt "${BASE_CKPT}" --source "${SOURCE_DATASET:-visa}" --eval_datasets "${EVAL_DATASETS:-mvtec,btad,mpdd}" \
  --source_root "${SOURCE_ROOT:-}" --target_root "${TARGET_ROOT:-}" --data_roots "${DATA_ROOTS_JSON}" \
  --seed "${SEED:-42}" --num_shards "${#GPUS[@]}" --val_fraction "${VAL_FRACTION:-0.2}" \
  --limit_per_category "${LIMIT_PER_CATEGORY:-0}" --image_size "${IMAGE_SIZE:-512}" --eval_size "${EVAL_SIZE:-512}" \
  --feature_grid "${FEATURE_GRID:-64}" --feature_dim "${FEATURE_DIM:-64}" \
  --tile_grid "${TILE_GRID:-2}" --tile_fraction "${TILE_FRACTION:-0.6}" --high_weight "${HIGH_WEIGHT:-0.25}" \
  --reference_pool "${REFERENCE_POOL:-4}" --reference_count "${REFERENCE_COUNT:-1}" \
  --match_topk "${MATCH_TOPK:-3}" --match_chunk "${MATCH_CHUNK:-128}" --match_temperature "${MATCH_TEMPERATURE:-0.07}" \
  --match_radius "${MATCH_RADIUS:-0.35}" --spatial_weight "${SPATIAL_WEIGHT:-0.15}" \
  --epochs "${HEAD_EPOCHS:-15}" --batch_size "${HEAD_BATCH_SIZE:-8}" --micro_batch "${MICRO_BATCH_SIZE:-2}" \
  --workers "${LOADER_WORKERS:-2}" --hidden "${HEAD_HIDDEN:-64}" --lr "${HEAD_LR:-0.0001}" \
  --precision "${PRECISION:-bf16}" --alphas "${REPAIR_ALPHAS:-0,0.25,0.5,1}" --val_fpr "${VAL_FPR:-0.01}" \
  --semantic_weight "${SEMANTIC_WEIGHT:-0.1}" --background_weight "${BACKGROUND_WEIGHT:-1}" --residual_weight "${RESIDUAL_WEIGHT:-0.01}" \
  --dino_repo_dir "${DINO_REPO_DIR:-./dinov3}" --dino_model_name "${DINO_MODEL_NAME:-dinov3_vitl16}" \
  --dino_weights "${DINO_WEIGHTS:-./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}" \
  --hfa_setting "${HFA_SETTING:-hfa3}" --dino_bottleneck "${DINO_BOTTLENECK:-256}" \
  --clip_model_name "${CLIP_MODEL_NAME:-ViT-L-14-336}" --clip_pretrained "${CLIP_PRETRAINED:-openai}" \
  --visual_layers "${VISUAL_LAYERS:-5,11,17,23}"
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_workers() {
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[${index}]}"; then
      unset 'PIDS[index]'
      echo 'Worker failed; subsequent stages will not start. Resume with the same WORK_DIR/config.' >&2
      return 1
    fi
    unset 'PIDS[index]'
  done
  PIDS=()
}
run_shards() {
  local stage="$1"
  PIDS=()
  for shard in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[${shard}]}" "${TRAIN_PYTHON}" reference_highres_experiment.py \
      --stage "${stage}" --work_dir "${WORK_DIR}" --shard_id "${shard}" &
    PIDS+=("$!")
  done
  wait_workers
}
if [[ "${RUN_EXPORT}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" reference_highres_experiment.py --stage preflight --work_dir "${WORK_DIR}"
  run_shards export
fi
"${TRAIN_PYTHON}" reference_highres_experiment.py --stage seal --work_dir "${WORK_DIR}"
if [[ "${RUN_TRAIN}" == 1 ]]; then
  MODES=(A_global B_highres C_reference D_highres_reference)
  # Independent same-budget heads, one per GPU. No DDP validation barriers.
  for ((start=0; start<${#MODES[@]}; start+=${#GPUS[@]})); do
    PIDS=()
    for slot in "${!GPUS[@]}"; do
      index=$((start+slot))
      ((index<${#MODES[@]})) || break
      CUDA_VISIBLE_DEVICES="${GPUS[${slot}]}" "${TRAIN_PYTHON}" reference_highres_experiment.py \
        --stage train --work_dir "${WORK_DIR}" --mode "${MODES[${index}]}" --until_epoch "${TRAIN_UNTIL_EPOCH:-5}" &
      PIDS+=("$!")
    done
    wait_workers
  done
fi
if [[ "${RUN_EVAL}" == 1 ]]; then
  run_shards evaluate
  "${TRAIN_PYTHON}" reference_highres_experiment.py --stage report --work_dir "${WORK_DIR}"
fi
echo "Done: ${WORK_DIR}/results/{README.md,summary.csv,comparisons.csv}; source_screen.csv"
echo 'Continue the SAME 15-epoch plan with TRAIN_UNTIL_EPOCH=15. Configuration/weights/code must stay unchanged.'
