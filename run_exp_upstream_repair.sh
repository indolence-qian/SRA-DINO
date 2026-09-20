#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_WORK_DIR="${SOURCE_WORK_DIR:-./checkpoint/upstream_localization_v1}"
WORK_DIR="${WORK_DIR:-./checkpoint/upstream_P0_P1_v1}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
RUN_P0="${RUN_P0:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
[[ -f "${SOURCE_WORK_DIR}/sealed.json" ]] || { echo 'SOURCE_WORK_DIR must contain a complete sealed v1 cache.' >&2; exit 1; }
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Bad GPU_IDS' >&2; exit 1; }
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[${gpu}]:-}" ]] || { echo 'Duplicate GPU id' >&2; exit 1; }
  SEEN[${gpu}]=1
done
[[ "${NPROC_PER_NODE:-${#GPUS[@]}}" == "${#GPUS[@]}" ]] || { echo 'NPROC_PER_NODE must match GPU_IDS' >&2; exit 1; }
for flag in "${RUN_P0}" "${RUN_TRAIN}" "${RUN_EVAL}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo 'Stage flags must be 0 or 1' >&2; exit 1; }
done
# Before even creating logs/locks, protect the read-only source directory.
"${TRAIN_PYTHON}" -c 'import sys; from pathlib import Path; s,w=map(lambda p:Path(p).resolve(),sys.argv[1:]); assert s!=w and not s.is_relative_to(w) and not w.is_relative_to(s), "Source/output must be separate non-nested directories"' "${SOURCE_WORK_DIR}" "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
command -v flock >/dev/null || { echo 'Install util-linux (flock)' >&2; exit 1; }
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo 'Output directory is already in use' >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
export PYTHONUNBUFFERED=1
echo "P0+P1: reuse ${SOURCE_WORK_DIR} READ ONLY; no Qwen/backbone reload; NO MARA"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c "import torch,cv2,sklearn; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; print(torch.__version__)"
"${TRAIN_PYTHON}" upstream_repair.py --stage prepare --work_dir "${WORK_DIR}" --source_work_dir "${SOURCE_WORK_DIR}" \
  --num_shards "${#GPUS[@]}" --epochs "${HEAD_EPOCHS:-15}" --batch_size "${HEAD_BATCH_SIZE:-4}" \
  --hidden "${HEAD_HIDDEN:-96}" --lr "${HEAD_LR:-0.0001}" --alphas "${REPAIR_ALPHAS:-0.1,0.25,0.5,1}" \
  --bg_weight "${BG_WEIGHT:-1}" --residual_weight "${RESIDUAL_WEIGHT:-0.01}" --hard_fraction "${HARD_BG_FRACTION:-0.01}" \
  --fpr "${VAL_FPR:-0.01}" --fpr_slack "${VAL_FPR_SLACK:-0.002}" --auc_tolerance "${VAL_AUC_TOLERANCE:-0.002}" \
  --val_pixels "${VAL_PIXELS:-8192}"
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_workers() {
  local stage="$1"
  PIDS=()
  for shard in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[${shard}]}" "${TRAIN_PYTHON}" upstream_repair.py \
      --stage "${stage}" --work_dir "${WORK_DIR}" --shard_id "${shard}" --workers "${LOADER_WORKERS:-2}" &
    PIDS+=("$!")
  done
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[${index}]}"; then
      unset 'PIDS[index]'
      echo "${stage} failed; stopping subsequent stages" >&2
      return 1
    fi
    unset 'PIDS[index]'
  done
  PIDS=()
}
if [[ "${RUN_P0}" == 1 ]]; then
  run_workers p0
  "${TRAIN_PYTHON}" upstream_repair.py --stage p0_report --work_dir "${WORK_DIR}"
fi
if [[ "${RUN_TRAIN}" == 1 ]]; then
  for mode in visual vlm; do
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc_per_node="${#GPUS[@]}" upstream_repair.py \
      --stage train --work_dir "${WORK_DIR}" --mode "${mode}" --workers "${LOADER_WORKERS:-2}"
  done
fi
if [[ "${RUN_EVAL}" == 1 ]]; then
  run_workers evaluate
  "${TRAIN_PYTHON}" upstream_repair.py --stage report --work_dir "${WORK_DIR}"
fi
echo "Done: ${WORK_DIR}/p0/metrics.csv and ${WORK_DIR}/results/metrics.csv"
echo 'alpha=0 / BASE_FALLBACK means no source-validated improvement; it is NOT a VLM gain.'
