#!/usr/bin/env bash
set -euo pipefail
# New supervised localization heads. Does NOT modify Base or run MARA.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
WORK_DIR="${WORK_DIR:-./checkpoint/upstream_localization_v1}"
BASE_CKPT="${BASE_CKPT:-}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
VLM_PYTHON="${VLM_PYTHON:-/root/miniconda3/envs/sra_vlm/bin/python}"
VLM_MODEL_ID="${VLM_MODEL_ID:-/mnt/qfg/Tate_qfg/models/Qwen3-VL-8B-Instruct-FP8}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
HEAD_MODES="${HEAD_MODES:-visual,vlm}"
RUN_REVIEW="${RUN_REVIEW:-1}"
RUN_EXPORT="${RUN_EXPORT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
[[ -e "${BASE_CKPT}" ]] || { echo 'Set BASE_CKPT to a frozen dual-tower Base file/directory.' >&2; exit 1; }
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU_IDS' >&2; exit 1; }
[[ "${HEAD_MODES}" == 'visual,vlm' || "${HEAD_MODES}" == 'visual' || "${HEAD_MODES}" == 'vlm' ]] || { echo 'HEAD_MODES=visual,vlm (recommended), visual or vlm' >&2; exit 1; }
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[${gpu}]:-}" ]] || { echo 'Duplicate GPU id' >&2; exit 1; }
  SEEN[${gpu}]=1
done
[[ "${NPROC_PER_NODE:-${#GPUS[@]}}" == "${#GPUS[@]}" ]] || { echo 'NPROC_PER_NODE must match GPU_IDS' >&2; exit 1; }
for flag in "${RUN_REVIEW}" "${RUN_EXPORT}" "${RUN_TRAIN}" "${RUN_EVAL}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo 'Stage flags must be 0/1' >&2; exit 1; }
done
mkdir -p "${WORK_DIR}"
command -v flock >/dev/null || { echo 'Install util-linux (flock).' >&2; exit 1; }
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo 'Another pipeline is using this WORK_DIR.' >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
export PYTHONUNBUFFERED=1
echo "Frozen dual tower + NEW dense head; modes=${HEAD_MODES}; NO MARA"
echo "Source=${SOURCE_DATASET:-visa} supervised TEST split -> source head train/val; targets=${EVAL_DATASETS:-mvtec}"
echo "GPUs=${GPU_IDS}, work=${WORK_DIR}; native full-coverage tiles, NO Base ROI gating"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c "import torch,numpy,PIL,cv2,sklearn; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; print('Train torch:',torch.__version__)"
if [[ "${RUN_REVIEW}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${VLM_PYTHON}" -c "import torch,vllm,qwen_vl_utils; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; print('VLM:',vllm.__version__)"
fi
"${TRAIN_PYTHON}" upstream_localization.py --stage prepare --work_dir "${WORK_DIR}" \
  --base_ckpt "${BASE_CKPT}" --model_id "${VLM_MODEL_ID}" \
  --source "${SOURCE_DATASET:-visa}" --eval_datasets "${EVAL_DATASETS:-mvtec}" \
  --source_root "${SOURCE_ROOT:-}" --target_root "${TARGET_ROOT:-}" \
  --val_fraction "${VAL_FRACTION:-0.2}" --limit_per_category "${LIMIT_PER_CATEGORY:-0}" \
  --seed "${SEED:-42}" --num_shards "${#GPUS[@]}" --image_size "${IMAGE_SIZE:-512}" \
  --tile_grid "${TILE_GRID:-3}" --tile_fraction "${TILE_FRACTION:-0.4}" \
  --gpu_memory "${VLM_GPU_MEMORY:-0.70}" --max_model_len "${VLM_MAX_MODEL_LEN:-4096}" \
  --max_tokens "${VLM_MAX_TOKENS:-256}" --teacher_image_size "${VLM_IMAGE_SIZE:-512}" \
  --retries "${VLM_RETRIES:-1}" --max_invalid_ratio "${VLM_MAX_INVALID_RATIO:-0.05}" \
  --dino_repo_dir "${DINO_REPO_DIR:-./dinov3}" --dino_model_name "${DINO_MODEL_NAME:-dinov3_vitl16}" \
  --dino_weights "${DINO_WEIGHTS:-./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}" \
  --hfa_setting "${HFA_SETTING:-hfa3}" --dino_bottleneck "${DINO_BOTTLENECK:-256}" \
  --clip_model_name "${CLIP_MODEL_NAME:-ViT-L-14-336}" --clip_pretrained "${CLIP_PRETRAINED:-openai}" \
  --visual_layers "${VISUAL_LAYERS:-5,11,17,23}"
# Fail on incompatible Base/DINO/CLIP dependencies before spending hours on VLM.
if [[ "${RUN_EXPORT}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" upstream_localization.py \
    --stage preflight --work_dir "${WORK_DIR}"
fi
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_workers() {
  local stage="$1" runtime="$2"
  PIDS=()
  for shard in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[${shard}]}" "${runtime}" upstream_localization.py \
      --stage "${stage}" --work_dir "${WORK_DIR}" --shard_id "${shard}" \
      --head_modes "${HEAD_MODES}" --batch_size "${HEAD_BATCH_SIZE:-4}" --workers "${LOADER_WORKERS:-2}" &
    PIDS+=("$!")
  done
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[${index}]}"; then
      unset 'PIDS[index]'
      echo "Failed ${stage}; no later stage will start. Rerun same command to resume." >&2
      return 1
    fi
    unset 'PIDS[index]'
  done
  PIDS=()
}
if [[ "${RUN_REVIEW}" == 1 ]]; then run_workers review "${VLM_PYTHON}"; fi
"${TRAIN_PYTHON}" upstream_localization.py --stage audit --work_dir "${WORK_DIR}"
if [[ "${RUN_EXPORT}" == 1 ]]; then run_workers export "${TRAIN_PYTHON}"; fi
"${TRAIN_PYTHON}" upstream_localization.py --stage seal --work_dir "${WORK_DIR}"
if [[ "${RUN_TRAIN}" == 1 ]]; then
  IFS=',' read -r -a MODES <<< "${HEAD_MODES}"
  for mode in "${MODES[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc_per_node="${#GPUS[@]}" \
      upstream_localization.py --stage train --work_dir "${WORK_DIR}" --mode "${mode}" \
      --epochs "${HEAD_EPOCHS:-15}" --batch_size "${HEAD_BATCH_SIZE:-4}" \
      --lr "${HEAD_LR:-0.0001}" --hidden "${HEAD_HIDDEN:-96}" --workers "${LOADER_WORKERS:-2}"
  done
fi
if [[ "${RUN_EVAL}" == 1 ]]; then
  run_workers evaluate "${TRAIN_PYTHON}"
  "${TRAIN_PYTHON}" upstream_localization.py --stage report --work_dir "${WORK_DIR}" --head_modes "${HEAD_MODES}"
fi
echo "Finished: ${WORK_DIR}/results/metrics.csv; review_audit.json; heads/{visual,vlm}/history.json"
