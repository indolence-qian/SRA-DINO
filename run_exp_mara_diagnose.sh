#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Frozen A/B only: no training, no VLM, no checkpoint auto-selection.
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
MARA_CKPT="${MARA_CKPT:-}"
WORK_DIR="${WORK_DIR:-./checkpoint/mara_early_stop_ab_v1}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
DATASETS="${DATASETS:-visa mvtec btad mpdd}"
DATA_ROOTS="${DATA_ROOTS:-}"
[[ -n "${DATA_ROOTS}" ]] || DATA_ROOTS='{}'
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-8}"
STEPS="${STEPS:-0}"
GRADIENT_IMAGES="${GRADIENT_IMAGES:-16}"
SEED="${SEED:-42}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_EXPORT="${RUN_EXPORT:-1}"
RUN_DIAGNOSE="${RUN_DIAGNOSE:-1}"
RUN_COLLECT="${RUN_COLLECT:-1}"

[[ -f "${MARA_CKPT}" ]] || {
  echo "[ERROR] MARA_CKPT must name one existing MARA .pth FILE, not a directory or Base checkpoint." >&2
  exit 1
}
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid GPU_IDS" >&2; exit 1; }
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[$gpu]:-}" ]] || { echo "Duplicate GPU id: ${gpu}" >&2; exit 1; }
  SEEN[$gpu]=1
done
for flag in "${RUN_EXPORT}" "${RUN_DIAGNOSE}" "${RUN_COLLECT}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo "Stage flags must be 0/1" >&2; exit 1; }
done
for budget in "${SAMPLES_PER_CATEGORY}" "${STEPS}" "${GRADIENT_IMAGES}" "${SEED}"; do
  [[ "${budget}" =~ ^[0-9]+$ ]] || { echo "Budgets/seed must be nonnegative integers" >&2; exit 1; }
done
[[ "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]] || { echo "CPU_THREADS must be positive" >&2; exit 1; }
command -v "${TRAIN_PYTHON}" >/dev/null || { echo "Missing TRAIN_PYTHON" >&2; exit 1; }
command -v flock >/dev/null || { echo "Missing flock (util-linux)" >&2; exit 1; }
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
"${TRAIN_PYTHON}" -c "import torch,numpy,cv2,PIL; print('Diagnostics dependencies:',torch.__version__)"
if [[ "${RUN_EXPORT}" == 1 ]]; then
  "${TRAIN_PYTHON}" -c "import torchvision,open_clip,sklearn,tqdm,test_mara; print('Existing Base environment ready')"
fi
if [[ "${RUN_EXPORT}" == 1 || "${RUN_DIAGNOSE}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c \
    "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}, 'Requested GPUs unavailable'; print('Visible GPUs:',torch.cuda.device_count())"
fi
[[ -d "${WORK_DIR}" ]] || mkdir -p "${WORK_DIR}"
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo "WORK_DIR is already running" >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1

EXTRA=()
[[ -z "${DINO_REPO_DIR:-}" ]] || EXTRA+=(--dino_repo_dir "${DINO_REPO_DIR}")
[[ -z "${DINO_WEIGHTS:-}" ]] || EXTRA+=(--dino_weights "${DINO_WEIGHTS}")
[[ -z "${CATEGORIES:-}" ]] || EXTRA+=(--categories "${CATEGORIES}")
echo "MARA A/B v2: one frozen checkpoint; shared Base caches; corrected no-op state semantics"
CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" mara_diagnose.py --stage prepare \
  --mara_ckpt "${MARA_CKPT}" --work_dir "${WORK_DIR}" \
  --datasets "${DATASETS}" --data_roots "${DATA_ROOTS}" \
  --samples_per_category "${SAMPLES_PER_CATEGORY}" --steps "${STEPS}" \
  --gradient_images "${GRADIENT_IMAGES}" --seed "${SEED}" \
  --num_shards "${#GPUS[@]}" --image_size "${IMAGE_SIZE:-512}" \
  --cpu_threads "${CPU_THREADS}" --pro_num_th "${PRO_NUM_TH:-1000}" \
  --pro_max_fpr "${PRO_MAX_FPR:-0.3}" --quality_tolerance "${QUALITY_TOLERANCE:-0.0001}" \
  --text_source "${TEXT_SOURCE:-prompt_learner}" \
  --clip_model_name "${CLIP_MODEL_NAME:-ViT-L-14-336}" \
  --clip_pretrained "${CLIP_PRETRAINED:-openai}" "${EXTRA[@]}"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_workers() {
  local stage="$1"
  PIDS=()
  for shard in "${!GPUS[@]}"; do
    echo "[$(date +'%F %T')] ${stage} shard=${shard}" >> "${WORK_DIR}/${stage}_shard_${shard}.log"
    CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "${TRAIN_PYTHON}" mara_diagnose.py \
      --stage "${stage}" --work_dir "${WORK_DIR}" --shard_id "${shard}" \
      --device cuda:0 --cpu_threads "${CPU_THREADS}" \
      >> "${WORK_DIR}/${stage}_shard_${shard}.log" 2>&1 &
    PIDS+=("$!")
  done
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$index]}"; then
      unset 'PIDS[index]'
      echo "[ERROR] ${stage} worker failed; inspect ${WORK_DIR}/${stage}_shard_*.log. Rerun unchanged command to resume." >&2
      exit 1
    fi
    unset 'PIDS[index]'
  done
}
if [[ "${RUN_EXPORT}" == 1 ]]; then
  echo "Export Base once/image, independent GPU workers (not DDP)"
  run_workers export
fi
if [[ "${RUN_DIAGNOSE}" == 1 ]]; then
  echo "A: step/return/STOP drift and frozen gradient audit; B: paired D0-D3"
  run_workers diagnose
fi
if [[ "${RUN_COLLECT}" == 1 ]]; then
  echo "CPU collection and category metrics; GPUs idle here is expected"
  CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" mara_diagnose.py --stage collect \
    --work_dir "${WORK_DIR}" --cpu_threads "${CPU_THREADS}"
fi
echo "Finished: ${WORK_DIR}/results/README.md"
