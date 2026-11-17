#!/usr/bin/env bash
# Reuse only locked predictions. No CUDA, training or inference required.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
WORK_DIR="${WORK_DIR:-./checkpoint/text_comparison_v1}"
REEVAL_DIR="${REEVAL_DIR:-${WORK_DIR}_reeval_v3}"
TRAIN_UNTIL_EPOCH="${TRAIN_UNTIL_EPOCH:-5}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"; TRAIN_SEEDS="${TRAIN_SEEDS//[[:space:]]/}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
RUN_EVAL="${RUN_EVAL:-1}"; UPDATE_REPO_REPORT="${UPDATE_REPO_REPORT:-1}"
[[ "${TRAIN_UNTIL_EPOCH}" =~ ^[1-9][0-9]*$ && "${EVAL_WORKERS}" =~ ^[1-9][0-9]*$ && "${TRAIN_SEEDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid horizon/seeds/workers' >&2; exit 1; }
for flag in "${RUN_EVAL}" "${UPDATE_REPO_REPORT}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo 'RUN_EVAL/UPDATE_REPO_REPORT must be 0/1' >&2; exit 1; }
done
command -v flock >/dev/null || { echo 'Install util-linux (flock)' >&2; exit 1; }
exec 8<"${WORK_DIR}/.run.lock"
flock -sn 8 || { echo 'Original text experiment is still running' >&2; exit 1; }
mkdir -p "${REEVAL_DIR}"
exec 9>"${REEVAL_DIR}/.run.lock"
flock -n 9 || { echo 'Another process owns REEVAL_DIR' >&2; exit 1; }
exec > >(tee -a "${REEVAL_DIR}/repair_$(date +'%Y%m%d_%H%M%S').log") 2>&1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPU_THREADS:-4}" MKL_NUM_THREADS="${CPU_THREADS:-4}"
DRIVER="${SCRIPT_DIR}/repair_text_evaluation.py"
COMMON=(--work_dir "${WORK_DIR}" --reeval_dir "${REEVAL_DIR}" --until_epoch "${TRAIN_UNTIL_EPOCH}" --train_seeds "${TRAIN_SEEDS}" --workers "${EVAL_WORKERS}")
echo 'Cached-prediction reevaluation: retain ALL images and as-provided masks, including officially released zero anomaly masks. NO subset exclusions.'
"${TRAIN_PYTHON}" "${DRIVER}" --stage prepare "${COMMON[@]}"
"${TRAIN_PYTHON}" "${DRIVER}" --stage audit "${COMMON[@]}"
PIDS=()
cleanup() { local pid; for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "${RUN_EVAL}" == 1 ]]; then
  for ((shard=0; shard<EVAL_WORKERS; shard++)); do
    "${TRAIN_PYTHON}" "${DRIVER}" --stage evaluate "${COMMON[@]}" --shard_id "${shard}" &
    PIDS+=("$!")
  done
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[${index}]}"; then
      unset 'PIDS[index]'; echo 'Evaluation worker failed; report will not be generated' >&2; exit 1
    fi
    unset 'PIDS[index]'
  done
  PIDS=()
  REPORT_ARGS=(); [[ "${UPDATE_REPO_REPORT}" == 0 ]] || REPORT_ARGS+=(--publish_report)
  "${TRAIN_PYTHON}" "${DRIVER}" --stage report "${COMMON[@]}" "${REPORT_ARGS[@]}"
fi
"${TRAIN_PYTHON}" "${DRIVER}" --stage package "${COMMON[@]}"
echo "Audit: ${REEVAL_DIR}/empty_masks.csv; report (when RUN_EVAL=1): ${REEVAL_DIR}/reports/对比实验.md"
