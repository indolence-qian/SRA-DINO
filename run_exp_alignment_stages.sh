#!/usr/bin/env bash
# H -> T -> locked frontend -> R -> target evaluation. Original v1 remains read-only.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
V1_DIR="${V1_DIR:-./checkpoint/reference_highres_v1}"
WORK_DIR="${WORK_DIR:-./checkpoint/alignment_three_stage_v2}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"
HEAD_EPOCHS="${HEAD_EPOCHS:-15}"
TRAIN_UNTIL_EPOCH="${TRAIN_UNTIL_EPOCH:-5}"
START_PHASE="${START_PHASE:-1}"
STOP_PHASE="${STOP_PHASE:-3}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_EXTRA_EXPORT="${RUN_EXTRA_EXPORT:-1}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
RUN_AUX0="${RUN_AUX0:-0}"
CONTINUE_ON_SOURCE_FALLBACK="${CONTINUE_ON_SOURCE_FALLBACK:-0}"
[[ -f "${V1_DIR}/config.json" && -f "${V1_DIR}/sealed.json" ]] || { echo 'V1_DIR must contain the original config, sealed cache, source masks and Base paths.'; exit 1; }
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ && "${TRAIN_SEEDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Use comma-separated distinct GPU ids and integer train seeds.'; exit 1; }
[[ "${START_PHASE}" =~ ^[123]$ && "${STOP_PHASE}" =~ ^[123]$ && "${START_PHASE}" -le "${STOP_PHASE}" ]] || { echo 'Require 1 <= START_PHASE <= STOP_PHASE <= 3.'; exit 1; }
for flag in "${RUN_TRAIN}" "${RUN_EVAL}" "${RUN_EXTRA_EXPORT}" "${RUN_BENCHMARK}" "${RUN_AUX0}" "${CONTINUE_ON_SOURCE_FALLBACK}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo 'Stage flags must be 0 or 1.'; exit 1; }
done
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
IFS=',' read -r -a SEEDS <<< "${TRAIN_SEEDS}"
declare -A SEEN=()
for item in "${GPUS[@]}"; do
  [[ -z "${SEEN[g${item}]:-}" ]] || { echo 'Duplicate GPU id'; exit 1; }
  SEEN[g${item}]=1
done
for item in "${SEEDS[@]}"; do
  [[ -z "${SEEN[s${item}]:-}" ]] || { echo 'Duplicate train seed'; exit 1; }
  SEEN[s${item}]=1
done
mkdir -p "${WORK_DIR}"
command -v flock >/dev/null || { echo 'Install util-linux (flock).'; exit 1; }
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo 'Another pipeline owns this WORK_DIR.'; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
export PYTHONUNBUFFERED=1
DRIVER="${SCRIPT_DIR}/alignment_stages_experiment.py"
COMMON=(--work_dir "${WORK_DIR}" --until_epoch "${TRAIN_UNTIL_EPOCH}")
FORCE=()
[[ "${CONTINUE_ON_SOURCE_FALLBACK}" == 0 ]] || FORCE=(--continue_on_fallback)
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c "import torch,numpy,scipy,sklearn,cv2,PIL; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; print('torch',torch.__version__)"
"${TRAIN_PYTHON}" "${DRIVER}" --stage prepare "${COMMON[@]}" --v1_dir "${V1_DIR}" \
  --epochs "${HEAD_EPOCHS}" --workers "${LOADER_WORKERS:-2}" --num_shards "${#GPUS[@]}" \
  --alphas "${REPAIR_ALPHAS:-0,0.05,0.1,0.25,0.5,1}" --keep_weight "${KEEP_WEIGHT:-0.1}"
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
      echo 'Worker failed: later stages will not start.'
      return 1
    fi
    unset 'PIDS[index]'
  done
}
train_modes() {
  local seed="$1"; shift
  local modes=("$@") offset slot index
  for ((offset=0; offset<${#modes[@]}; offset+=${#GPUS[@]})); do
    PIDS=()
    for slot in "${!GPUS[@]}"; do
      index=$((offset+slot))
      [[ "${index}" -lt "${#modes[@]}" ]] || break
      CUDA_VISIBLE_DEVICES="${GPUS[${slot}]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage train "${COMMON[@]}" \
        --train_seed "${seed}" --mode "${modes[${index}]}" &
      PIDS+=("$!")
    done
    wait_workers
  done
}
export_partition() {
  local partition="$1" slot
  if [[ "${RUN_EXTRA_EXPORT}" == 1 ]]; then
    # The first real query must pass parity against the old sealed global/local evidence.
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage export "${COMMON[@]}" --partition "${partition}" --shard_id 0 --probe
    PIDS=()
    for slot in "${!GPUS[@]}"; do
      CUDA_VISIBLE_DEVICES="${GPUS[${slot}]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage export "${COMMON[@]}" --partition "${partition}" --shard_id "${slot}" &
      PIDS+=("$!")
    done
    wait_workers
  fi
  "${TRAIN_PYTHON}" "${DRIVER}" --stage seal "${COMMON[@]}" --partition "${partition}"
}
SOURCE_EXPORTED=0
NEEDS_TARGET_EXTRA=0
for seed in "${SEEDS[@]}"; do
  if [[ "${RUN_TRAIN}" == 1 && "${START_PHASE}" -le 1 ]]; then
    MODES=(H1 H2 H3)
    [[ "${RUN_AUX0}" == 0 ]] || MODES+=(H1_aux0)
    train_modes "${seed}" "${MODES[@]}"
  fi
  ALLOWED=0
  if "${TRAIN_PYTHON}" "${DRIVER}" --stage gate "${COMMON[@]}" --train_seed "${seed}" "${FORCE[@]}"; then
    ALLOWED=1
  else
    code=$?; [[ "${code}" == 3 ]] || exit "${code}"
  fi
  if [[ "${STOP_PHASE}" -ge 2 && "${ALLOWED}" == 1 ]]; then
    NEEDS_TARGET_EXTRA=1
    if [[ "${RUN_TRAIN}" == 1 && "${START_PHASE}" -le 2 ]]; then
      if [[ "${SOURCE_EXPORTED}" == 0 ]]; then export_partition source; SOURCE_EXPORTED=1; fi
      train_modes "${seed}" T10 T01 T11
    fi
  fi
  "${TRAIN_PYTHON}" "${DRIVER}" --stage select "${COMMON[@]}" --train_seed "${seed}" --front_level "${STOP_PHASE}"
  if [[ "${STOP_PHASE}" -ge 3 ]]; then
    "${TRAIN_PYTHON}" "${DRIVER}" --stage fit_distance "${COMMON[@]}"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage calibrate_r1 "${COMMON[@]}" --train_seed "${seed}"
    if "${TRAIN_PYTHON}" "${DRIVER}" --stage reference_gate "${COMMON[@]}" --train_seed "${seed}" "${FORCE[@]}"; then
      if [[ "${RUN_TRAIN}" == 1 ]]; then train_modes "${seed}" R2 R3; fi
    else
      code=$?; [[ "${code}" == 3 ]] || exit "${code}"
      echo 'No eligible learned frontend: R1 diagnosis kept, R2/R3 intentionally skipped.'
    fi
  fi
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage audit "${COMMON[@]}" --train_seed "${seed}" --front_level "${STOP_PHASE}"
done
if [[ "${RUN_EVAL}" == 1 ]]; then
  if [[ "${NEEDS_TARGET_EXTRA}" == 1 ]]; then export_partition eval; fi
  for seed in "${SEEDS[@]}"; do
    PIDS=()
    for slot in "${!GPUS[@]}"; do
      CUDA_VISIBLE_DEVICES="${GPUS[${slot}]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage evaluate "${COMMON[@]}" \
        --train_seed "${seed}" --front_level "${STOP_PHASE}" --shard_id "${slot}" &
      PIDS+=("$!")
    done
    wait_workers
  done
  "${TRAIN_PYTHON}" "${DRIVER}" --stage report "${COMMON[@]}" --train_seeds "${TRAIN_SEEDS}" --front_level "${STOP_PHASE}"
  if [[ "${RUN_BENCHMARK}" == 1 ]]; then
    for seed in "${SEEDS[@]}"; do
      CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${TRAIN_PYTHON}" "${DRIVER}" --stage benchmark "${COMMON[@]}" \
        --train_seed "${seed}" --front_level "${STOP_PHASE}" --benchmark_images "${BENCHMARK_IMAGES:-4}"
    done
  fi
fi
echo "Completed through phase ${STOP_PHASE}; artifacts in ${WORK_DIR}. Target reports exist only when RUN_EVAL=1."
