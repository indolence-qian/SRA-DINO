#!/usr/bin/env bash
# Frozen residual features/heads -> source lambda selection -> targets -> HTML.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
TEXT_WORK_DIR="${TEXT_WORK_DIR:-./checkpoint/text_comparison_v1}"
WORK_DIR="${WORK_DIR:-./checkpoint/text_lambda_v1_through15}"
TRAIN_UNTIL_EPOCH="${TRAIN_UNTIL_EPOCH:-15}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"; TRAIN_SEEDS="${TRAIN_SEEDS//[[:space:]]/}"
GPU_IDS="${GPU_IDS:-0,1}"; GPU_IDS="${GPU_IDS//[[:space:]]/}"
RUN_EXPORT="${RUN_EXPORT:-1}"; RUN_EVAL="${RUN_EVAL:-1}"; UPDATE_REPO_REPORT="${UPDATE_REPO_REPORT:-1}"
EVAL_PARALLEL="${EVAL_PARALLEL:-1}"
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ && "${TRAIN_SEEDS}" =~ ^[0-9]+(,[0-9]+)*$ && "${TRAIN_UNTIL_EPOCH}" =~ ^[1-9][0-9]*$ && "${EVAL_PARALLEL}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid GPU_IDS/seeds/horizon/EVAL_PARALLEL' >&2; exit 1; }
for flag in "${RUN_EXPORT}" "${RUN_EVAL}" "${UPDATE_REPO_REPORT}"; do [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo 'RUN_* flags must be 0/1' >&2; exit 1; }; done
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
IFS=',' read -r -a SEEDS <<< "${TRAIN_SEEDS}"
declare -A SEEN=()
for value in "${GPUS[@]}"; do [[ -z "${SEEN[g${value}]:-}" ]] || { echo 'Duplicate GPU' >&2; exit 1; }; SEEN[g${value}]=1; done
for value in "${SEEDS[@]}"; do [[ -z "${SEEN[s${value}]:-}" ]] || { echo 'Duplicate seed' >&2; exit 1; }; SEEN[s${value}]=1; done
command -v flock >/dev/null || { echo 'Install util-linux (flock)' >&2; exit 1; }
exec 8<"${TEXT_WORK_DIR}/.run.lock"
flock -sn 8 || { echo 'Original text experiment is still running' >&2; exit 1; }
mkdir -p "${WORK_DIR}"
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo 'Another process owns this WORK_DIR' >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPU_THREADS:-4}" MKL_NUM_THREADS="${CPU_THREADS:-4}"
echo 'No training, no DINO export. Cache V/A from frozen residual heads once; source-only lambda selection; all target masks retained.'
if [[ "${RUN_EXPORT}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c "import torch,numpy,PIL,scipy,sklearn,cv2; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; print('torch',torch.__version__)"
fi
DRIVER="${SCRIPT_DIR}/lambda_text_experiment.py"
COMMON=(--work_dir "${WORK_DIR}")
"${TRAIN_PYTHON}" "${DRIVER}" --stage prepare "${COMMON[@]}" --text_work_dir "${TEXT_WORK_DIR}" \
  --until_epoch "${TRAIN_UNTIL_EPOCH}" --train_seeds "${TRAIN_SEEDS}" --num_shards "${#GPUS[@]}" --html_map_side "${HTML_MAP_SIDE:-96}"
PIDS=()
cleanup() { local pid; for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_workers() {
  local index
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[${index}]}"; then
      unset 'PIDS[index]'; echo 'Worker failed; later evaluation/HTML stages will not start. Resume with the same settings.' >&2; return 1
    fi
    unset 'PIDS[index]'
  done
  PIDS=()
}
export_partition() {
  local partition="$1" seed slot
  for seed in "${SEEDS[@]}"; do
    if [[ "${RUN_EXPORT}" == 1 ]]; then
      PIDS=()
      for slot in "${!GPUS[@]}"; do
        CUDA_VISIBLE_DEVICES="${GPUS[${slot}]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage export "${COMMON[@]}" --partition "${partition}" --train_seed "${seed}" --shard_id "${slot}" &
        PIDS+=("$!")
      done
      wait_workers
    fi
    "${TRAIN_PYTHON}" "${DRIVER}" --stage seal "${COMMON[@]}" --partition "${partition}" --train_seed "${seed}"
  done
}
export_partition source
"${TRAIN_PYTHON}" "${DRIVER}" --stage select "${COMMON[@]}"
export_partition eval
for seed in "${SEEDS[@]}"; do "${TRAIN_PYTHON}" "${DRIVER}" --stage lock_predictions "${COMMON[@]}" --train_seed "${seed}"; done
if [[ "${RUN_EVAL}" == 1 ]]; then
  # Evaluation is CPU-heavy; limit concurrent ROC sorts independently of GPU count.
  for ((offset=0; offset<${#GPUS[@]}; offset+=EVAL_PARALLEL)); do
    PIDS=()
    for ((slot=offset; slot<offset+EVAL_PARALLEL && slot<${#GPUS[@]}; slot++)); do
      "${TRAIN_PYTHON}" "${DRIVER}" --stage evaluate "${COMMON[@]}" --shard_id "${slot}" &
      PIDS+=("$!")
    done
    wait_workers
  done
fi
REPORT_ARGS=(); [[ "${UPDATE_REPO_REPORT}" == 0 ]] || REPORT_ARGS+=(--publish_report)
"${TRAIN_PYTHON}" "${DRIVER}" --stage report "${COMMON[@]}" "${REPORT_ARGS[@]}"
"${TRAIN_PYTHON}" "${DRIVER}" --stage package "${COMMON[@]}"
echo "Completed: ${WORK_DIR}/reports/lambda_comparison.md; ${WORK_DIR}/html/index.html; lambda_text_html_resources.zip"
