#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Experiment G is inference-only. Each GPU owns independent complete categories.
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
MARA_CKPT="${MARA_CKPT:-}"
WORK_DIR="${WORK_DIR:-./checkpoint/mara_gate_sweep_g_v1}"
SOURCE_WORK_DIR="${SOURCE_WORK_DIR:-}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
GATE_MARGINS="${GATE_MARGINS:-0,-0.01,-0.03,-0.05,-0.10}"
DATASETS="${DATASETS:-visa mvtec btad mpdd}"
DATA_ROOTS="${DATA_ROOTS:-}"
[[ -n "${DATA_ROOTS}" ]] || DATA_ROOTS='{}'
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-8}"
STEPS="${STEPS:-0}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-8}"
SWEEP_BATCH_SIZE="${SWEEP_BATCH_SIZE:-8}"
DECODE_WORKERS="${DECODE_WORKERS:-2}"
METRIC_WORKERS="${METRIC_WORKERS:-2}"
CPU_THREADS="${CPU_THREADS:-4}"
SEED="${SEED:-42}"
EMPTY_GT_POLICY="${EMPTY_GT_POLICY:-exclude}"
[[ -f "${MARA_CKPT}" ]] || { echo "[ERROR] MARA_CKPT must be an existing MARA .pth FILE" >&2; exit 1; }
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid GPU_IDS" >&2; exit 1; }
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[$gpu]:-}" ]] || { echo "Duplicate GPU id ${gpu}" >&2; exit 1; }
  SEEN[$gpu]=1
done
for positive in "${EXPORT_BATCH_SIZE}" "${SWEEP_BATCH_SIZE}" "${DECODE_WORKERS}" "${METRIC_WORKERS}" "${CPU_THREADS}"; do
  [[ "${positive}" =~ ^[1-9][0-9]*$ ]] || { echo "Batch/thread budgets must be positive integers" >&2; exit 1; }
done
for budget in "${SAMPLES_PER_CATEGORY}" "${STEPS}" "${SEED}"; do
  [[ "${budget}" =~ ^[0-9]+$ ]] || { echo "Invalid nonnegative budget" >&2; exit 1; }
done
[[ "${EMPTY_GT_POLICY}" == exclude || "${EMPTY_GT_POLICY}" == flag ]] || { echo "Invalid EMPTY_GT_POLICY" >&2; exit 1; }
command -v "${TRAIN_PYTHON}" >/dev/null
command -v flock >/dev/null
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}"
"${TRAIN_PYTHON}" -c "import torch,numpy,cv2,PIL; print('G dependencies:',torch.__version__)"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c \
  "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}, 'Requested GPUs unavailable'; print([(torch.cuda.get_device_name(i),round(torch.cuda.get_device_properties(i).total_memory/2**30,1)) for i in range(torch.cuda.device_count())])"
mkdir -p "${WORK_DIR}"
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo "WORK_DIR is already running" >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
EXTRA=()
[[ -z "${SOURCE_WORK_DIR}" ]] || EXTRA+=(--source_work_dir "${SOURCE_WORK_DIR}")
[[ -z "${DINO_REPO_DIR:-}" ]] || EXTRA+=(--dino_repo_dir "${DINO_REPO_DIR}")
[[ -z "${DINO_WEIGHTS:-}" ]] || EXTRA+=(--dino_weights "${DINO_WEIGHTS}")
[[ -z "${CATEGORIES:-}" ]] || EXTRA+=(--categories "${CATEGORIES}")
echo "Experiment G: frozen checkpoint, FP32, margins=${GATE_MARGINS}; no training/no VLM"
CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" mara_gate_sweep.py --stage prepare \
  --work_dir "${WORK_DIR}" --mara_ckpt "${MARA_CKPT}" --datasets "${DATASETS}" --data_roots "${DATA_ROOTS}" \
  --samples_per_category "${SAMPLES_PER_CATEGORY}" --steps "${STEPS}" --seed "${SEED}" \
  --num_shards "${#GPUS[@]}" --gate_margins "${GATE_MARGINS}" --empty_gt_policy "${EMPTY_GT_POLICY}" \
  --export_batch_size "${EXPORT_BATCH_SIZE}" --sweep_batch_size "${SWEEP_BATCH_SIZE}" \
  --decode_workers "${DECODE_WORKERS}" --metric_workers "${METRIC_WORKERS}" --cpu_threads "${CPU_THREADS}" \
  --image_size "${IMAGE_SIZE:-512}" --pro_num_th "${PRO_NUM_TH:-1000}" --pro_max_fpr "${PRO_MAX_FPR:-0.3}" \
  --quality_tolerance "${QUALITY_TOLERANCE:-0.0001}" --text_source "${TEXT_SOURCE:-prompt_learner}" \
  --clip_model_name "${CLIP_MODEL_NAME:-ViT-L-14-336}" --clip_pretrained "${CLIP_PRETRAINED:-openai}" "${EXTRA[@]}"
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_workers() {
  local stage="$1"
  PIDS=()
  for shard in "${!GPUS[@]}"; do
    local log="${WORK_DIR}/${stage}_shard_${shard}.log"
    echo "[$(date +'%F %T')] ${stage} GPU=${GPUS[$shard]} shard=${shard}" >> "${log}"
    echo "Launching ${stage} GPU=${GPUS[$shard]}; log=${log}"
    CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "${TRAIN_PYTHON}" mara_gate_sweep.py \
      --stage "${stage}" --work_dir "${WORK_DIR}" --shard_id "${shard}" --device cuda:0 \
      --cpu_threads "${CPU_THREADS}" >> "${log}" 2>&1 &
    PIDS+=("$!")
  done
  # Explicit PIDs also handle workers that finish before the first wait call.
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$index]}"; then
      unset 'PIDS[index]'
      echo "[ERROR] ${stage} worker failed. Log tails follow:" >&2
      for shard in "${!GPUS[@]}"; do
        echo "===== ${stage} shard=${shard} =====" >&2
        tail -n 70 "${WORK_DIR}/${stage}_shard_${shard}.log" >&2
      done
      exit 1
    fi
    unset 'PIDS[index]'
  done
  PIDS=()
}
run_workers export
CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" mara_gate_sweep.py --stage audit --work_dir "${WORK_DIR}" --cpu_threads "${CPU_THREADS}"
run_workers sweep
CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" mara_gate_sweep.py --stage collect --work_dir "${WORK_DIR}" --cpu_threads "${CPU_THREADS}"
echo "Finished. Upload only: ${WORK_DIR}/gate_report.zip"
