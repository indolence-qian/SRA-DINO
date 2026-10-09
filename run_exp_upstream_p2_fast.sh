#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
SOURCE_WORK_DIR="${SOURCE_WORK_DIR:-./checkpoint/upstream_localization_v1}"
WORK_DIR="${WORK_DIR:-./checkpoint/upstream_P2_fast_v1}"
DIAGNOSTIC_WORK_DIR="${DIAGNOSTIC_WORK_DIR:-./checkpoint/upstream_P2_v1}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
[[ -f "${SOURCE_WORK_DIR}/sealed.json" ]] || { echo 'Original sealed localization cache is required' >&2; exit 1; }
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Bad GPU_IDS' >&2; exit 1; }
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[${gpu}]:-}" ]] || { echo 'Duplicate GPU id' >&2; exit 1; }
  SEEN[${gpu}]=1
done
for flag in "${RUN_TRAIN}" "${RUN_EVAL}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo 'Stage flags must be 0 or 1' >&2; exit 1; }
done
"${TRAIN_PYTHON}" -c 'import sys; from pathlib import Path; s,w=map(lambda p:Path(p).resolve(),sys.argv[1:]); assert s!=w and not s.is_relative_to(w) and not w.is_relative_to(s), "Source/output must be separate non-nested directories"' "${SOURCE_WORK_DIR}" "${WORK_DIR}"
command -v flock >/dev/null || { echo 'Install util-linux (flock)' >&2; exit 1; }
mkdir -p "${WORK_DIR}"
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo 'Output directory is already in use' >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CPU_THREADS:-4}"
export MKL_NUM_THREADS="${CPU_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS:-4}"
"${TRAIN_PYTHON}" -c "import torch,cv2,sklearn,matplotlib; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; print(torch.__version__)"
echo "Fast P2: ${#GPUS[@]} independent GPU workers; planned=${HEAD_EPOCHS:-15}, run_until=${TRAIN_UNTIL_EPOCH:-5}; no Qwen/backbone reload"
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_stage() {
  "${TRAIN_PYTHON}" upstream_p2_fast.py --stage "$1" --work_dir "${WORK_DIR}" \
    --until_epoch "${TRAIN_UNTIL_EPOCH:-5}" --micro_batch "${MICRO_BATCH_SIZE:-8}" \
    --reserve_gb "${GPU_RESERVE_GB:-4}" --gpu_cache_gb "${GPU_CACHE_GB:-0}" \
    --cpu_threads "${CPU_THREADS:-4}" --pack_workers "${PACK_WORKERS:-4}" \
    --diagnostic_work_dir "${DIAGNOSTIC_WORK_DIR}" &
  PIDS=("$!")
  if ! wait "${PIDS[0]}"; then
    echo "Stage $1 failed; stopping subsequent stages" >&2
    return 1
  fi
  PIDS=()
}
"${TRAIN_PYTHON}" upstream_p2_fast.py --stage prepare --work_dir "${WORK_DIR}" --source_work_dir "${SOURCE_WORK_DIR}" \
  --num_shards "${#GPUS[@]}" --epochs "${HEAD_EPOCHS:-15}" --until_epoch "${TRAIN_UNTIL_EPOCH:-5}" \
  --batch_size "${EFFECTIVE_BATCH_SIZE:-8}" --micro_batch "${MICRO_BATCH_SIZE:-8}" \
  --hidden "${HEAD_HIDDEN:-96}" --lr "${HEAD_LR:-0.0001}" --precision "${HEAD_PRECISION:-bf16}" \
  --alphas "${REPAIR_ALPHAS:-0.1,0.25,0.5,1}" --val_pixels "${VAL_PIXELS:-8192}" \
  --bg_weight "${BG_WEIGHT:-1}" --residual_weight "${RESIDUAL_WEIGHT:-0.01}" --hard_fraction "${HARD_BG_FRACTION:-0.01}" \
  --fpr "${VAL_FPR:-0.01}" --fpr_slack "${VAL_FPR_SLACK:-0.002}" --auc_tolerance "${VAL_AUC_TOLERANCE:-0.002}" \
  --component_weight "${COMPONENT_WEIGHT:-0.1}" --ranking_weight "${RANKING_WEIGHT:-0.1}" \
  --small_fraction "${SMALL_FRACTION:-0.001}" --small_weight "${SMALL_WEIGHT:-2}" --min_area "${MIN_COMPONENT_AREA:-2}" \
  --near_radius "${NEAR_RADIUS:-8}" --rank_pixels "${RANK_PIXELS:-64}" --rank_margin "${RANK_MARGIN:-1}"
run_stage diagnostics
run_stage pack
if [[ "${RUN_TRAIN}" == 1 ]]; then run_stage train; fi
if [[ "${RUN_EVAL}" == 1 ]]; then
  run_stage evaluate
  run_stage report
fi
echo "Done: ${WORK_DIR}/source_screen.csv and ${WORK_DIR}/results/README.md"
echo 'To continue the SAME run to 15 epochs, keep config/WORK_DIR and set TRAIN_UNTIL_EPOCH=15.'
