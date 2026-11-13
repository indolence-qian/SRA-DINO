#!/usr/bin/env bash
# Read-only factorial D -> matched T1/T2/T3 heads -> locked evaluation/report.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
BASE_WORK_DIR="${BASE_WORK_DIR:-./checkpoint/factorial_hr_residual_v1}"
WORK_DIR="${WORK_DIR:-./checkpoint/text_comparison_v1}"
GPU_IDS="${GPU_IDS:-0,1}"; GPU_IDS="${GPU_IDS//[[:space:]]/}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"; TRAIN_SEEDS="${TRAIN_SEEDS//[[:space:]]/}"
BASE_TRAIN_SEED="${BASE_TRAIN_SEED:-42}"; BASE_UNTIL_EPOCH="${BASE_UNTIL_EPOCH:-5}"
EPOCHS="${EPOCHS:-15}"; TRAIN_UNTIL_EPOCH="${TRAIN_UNTIL_EPOCH:-5}"
RUN_TRAIN="${RUN_TRAIN:-1}"; RUN_PREDICT="${RUN_PREDICT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"; RUN_PACKAGE="${RUN_PACKAGE:-1}"
UPDATE_REPO_REPORT="${UPDATE_REPO_REPORT:-1}"
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ && "${TRAIN_SEEDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU_IDS/TRAIN_SEEDS' >&2; exit 1; }
[[ "${BASE_TRAIN_SEED}" =~ ^[0-9]+$ && "${BASE_UNTIL_EPOCH}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid BASE_TRAIN_SEED/BASE_UNTIL_EPOCH' >&2; exit 1; }
[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ && "${TRAIN_UNTIL_EPOCH}" =~ ^[1-9][0-9]*$ && "${TRAIN_UNTIL_EPOCH}" -le "${EPOCHS}" ]] || { echo 'Require 1 <= TRAIN_UNTIL_EPOCH <= EPOCHS' >&2; exit 1; }
for flag in "${RUN_TRAIN}" "${RUN_PREDICT}" "${RUN_EVAL}" "${RUN_PACKAGE}" "${UPDATE_REPO_REPORT}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo 'RUN_* and UPDATE_REPO_REPORT must be 0/1' >&2; exit 1; }
done
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
IFS=',' read -r -a SEEDS <<< "${TRAIN_SEEDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[g${gpu}]:-}" ]] || { echo 'Duplicate GPU' >&2; exit 1; }; SEEN[g${gpu}]=1
done
for seed in "${SEEDS[@]}"; do
  [[ -z "${SEEN[s${seed}]:-}" ]] || { echo 'Duplicate seed' >&2; exit 1; }; SEEN[s${seed}]=1
done
mkdir -p "${WORK_DIR}"
command -v flock >/dev/null || { echo 'Install util-linux (flock)' >&2; exit 1; }
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo 'Another process owns this WORK_DIR' >&2; exit 1; }
# Coordinate with the first runner so it cannot change the parent during this run.
[[ -d "${BASE_WORK_DIR}" ]] || { echo 'BASE_WORK_DIR must contain the completed first factorial experiment' >&2; exit 1; }
exec 8<"${BASE_WORK_DIR}/.run.lock"
flock -sn 8 || { echo 'Parent factorial experiment is running; finish it before this comparison' >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPU_THREADS:-4}" MKL_NUM_THREADS="${CPU_THREADS:-4}"
echo 'T1=single paired delta, T2=two ordinary state logits, T3=mean paired-delta scores; one SHARED frozen D/Flow.'
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c \
  "import torch,numpy,PIL,scipy,sklearn,cv2; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; print('torch',torch.__version__)"
DRIVER="${SCRIPT_DIR}/text_alignment_ablation.py"
COMMON=(--work_dir "${WORK_DIR}" --until_epoch "${TRAIN_UNTIL_EPOCH}")
OVERRIDES=()
[[ -z "${BATCH_SIZE:-}" ]] || OVERRIDES+=(--batch_size "${BATCH_SIZE}")
[[ -z "${MICRO_BATCH_SIZE:-}" ]] || OVERRIDES+=(--micro_batch "${MICRO_BATCH_SIZE}")
[[ -z "${LOADER_WORKERS:-}" ]] || OVERRIDES+=(--workers "${LOADER_WORKERS}")
[[ -z "${TEXT_LR:-}" ]] || OVERRIDES+=(--text_lr "${TEXT_LR}")
"${TRAIN_PYTHON}" "${DRIVER}" --stage prepare "${COMMON[@]}" \
  --base_work_dir "${BASE_WORK_DIR}" --base_train_seed "${BASE_TRAIN_SEED}" --base_until_epoch "${BASE_UNTIL_EPOCH}" \
  --epochs "${EPOCHS}" --num_shards "${#GPUS[@]}" --text_strength "${TEXT_STRENGTH:-0.5}" \
  --prompt_file "${PROMPT_FILE:-${SCRIPT_DIR}/configs/text_ablation_prompts_v1.json}" "${OVERRIDES[@]}"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage text_bank "${COMMON[@]}"
PIDS=()
cleanup() { local pid; for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_workers() {
  local index
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[${index}]}"; then
      unset 'PIDS[index]'; echo 'Worker failed: later stages will not start. Resume with the same settings.' >&2; return 1
    fi
    unset 'PIDS[index]'
  done
  PIDS=()
}
train_texts() {
  local seed="$1" offset slot index
  local arms=(T1 T2 T3)
  for ((offset=0; offset<3; offset+=${#GPUS[@]})); do
    PIDS=()
    for slot in "${!GPUS[@]}"; do
      index=$((offset+slot)); [[ "${index}" -lt 3 ]] || break
      CUDA_VISIBLE_DEVICES="${GPUS[${slot}]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage train "${COMMON[@]}" \
        --train_seed "${seed}" --arm "${arms[${index}]}" &
      PIDS+=("$!")
    done
    wait_workers
  done
}
run_shards() {
  local stage="$1" seed="$2" slot
  PIDS=()
  for slot in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[${slot}]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage "${stage}" "${COMMON[@]}" \
      --shard_id "${slot}" --train_seed "${seed}" &
    PIDS+=("$!")
  done
  wait_workers
}
for seed in "${SEEDS[@]}"; do
  if [[ "${RUN_TRAIN}" == 1 ]]; then train_texts "${seed}"; fi
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage select "${COMMON[@]}" --train_seed "${seed}"
done
# All source training/selection finishes before target inference/evaluation.
if [[ "${RUN_PREDICT}" == 1 || "${RUN_EVAL}" == 1 ]]; then
  for seed in "${SEEDS[@]}"; do
    if [[ "${RUN_PREDICT}" == 1 ]]; then run_shards predict "${seed}"; fi
    "${TRAIN_PYTHON}" "${DRIVER}" --stage lock_predictions "${COMMON[@]}" --train_seed "${seed}"
    if [[ "${RUN_EVAL}" == 1 ]]; then run_shards evaluate "${seed}"; fi
  done
fi
if [[ "${RUN_EVAL}" == 1 ]]; then
  REPORT_ARGS=(); [[ "${UPDATE_REPO_REPORT}" == 0 ]] || REPORT_ARGS+=(--publish_report)
  "${TRAIN_PYTHON}" "${DRIVER}" --stage report "${COMMON[@]}" --train_seeds "${TRAIN_SEEDS}" "${REPORT_ARGS[@]}"
fi
if [[ "${RUN_PACKAGE}" == 1 ]]; then "${TRAIN_PYTHON}" "${DRIVER}" --stage package "${COMMON[@]}"; fi
echo "Completed. Results: ${WORK_DIR}/reports/through${TRAIN_UNTIL_EPOCH}/对比实验.md"
