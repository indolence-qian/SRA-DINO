#!/usr/bin/env bash
set -euo pipefail
if (( BASH_VERSINFO[0] < 4 )); then
  echo "[ERROR] Bash 4+ required" >&2
  exit 1
fi
# Existing local export -> input audit -> A/B/C/D review -> image probes -> audit.
# No training, no new Base export, no changes to source cache or formal detector.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
SOURCE_WORK_DIR="${SOURCE_WORK_DIR:-./checkpoint/dual_vlm_local_visa_v1}"
WORK_DIR="${WORK_DIR:-./checkpoint/vlm_diagnose_visa_v1}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
VLM_PYTHON="${VLM_PYTHON:-/root/miniconda3/envs/sra_vlm/bin/python}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
RUN_REVIEW="${RUN_REVIEW:-1}"
RUN_SANITY="${RUN_SANITY:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "[ERROR] Invalid GPU_IDS" >&2; exit 1; }
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[${gpu}]:-}" ]] || { echo "[ERROR] Duplicate GPU id" >&2; exit 1; }
  SEEN[${gpu}]=1
done
[[ -z "${NPROC_PER_NODE:-}" || "${NPROC_PER_NODE}" == "${#GPUS[@]}" ]] || { echo "[ERROR] NPROC_PER_NODE must match GPU_IDS" >&2; exit 1; }
for flag in "${RUN_REVIEW}" "${RUN_SANITY}" "${RUN_EVAL}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo "[ERROR] Stage flags must be 0/1" >&2; exit 1; }
done
[[ -f "${SOURCE_WORK_DIR}/manifest.json" && -f "${SOURCE_WORK_DIR}/config.json" ]] || {
  echo "[ERROR] SOURCE_WORK_DIR must contain the previous local experiment manifest.json/config.json and crops." >&2
  exit 1
}
command -v flock >/dev/null || { echo "[ERROR] Install util-linux for flock" >&2; exit 1; }
mkdir -p "${WORK_DIR}"
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo "[ERROR] Diagnostic directory is already in use" >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
export PYTHONUNBUFFERED=1
echo "VLM diagnosis only: source=${SOURCE_WORK_DIR}, output=${WORK_DIR}, GPUs=${GPU_IDS}"
"${TRAIN_PYTHON}" -c "import numpy,PIL,cv2; print('CPU audit dependencies ready')"
if [[ "${RUN_REVIEW}" == 1 || "${RUN_SANITY}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${VLM_PYTHON}" -c "import os,shutil,torch,vllm,qwen_vl_utils; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; assert shutil.which(os.environ.get('CC','gcc')) or (not os.environ.get('CC') and shutil.which('clang')), 'Install build-essential'; print('VLM:',torch.__version__,vllm.__version__)"
fi
MODEL_FLAGS=()
[[ -z "${VLM_MODEL_ID:-}" ]] || MODEL_FLAGS=(--model_id "${VLM_MODEL_ID}")
CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" vlm_diagnose.py --stage prepare \
  --source_dir "${SOURCE_WORK_DIR}" --work_dir "${WORK_DIR}" \
  --candidates "${DIAG_CANDIDATES:-96}" --seed "${SEED:-42}" \
  --num_shards "${#GPUS[@]}" --min_pixels "${DIAG_MIN_PIXELS:-65536}" \
  --sanity_samples "${DIAG_SANITY_SAMPLES:-6}" "${MODEL_FLAGS[@]}"
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_workers() {
  local stage="$1" variant="$2"
  PIDS=()
  for shard in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[${shard}]}" "${VLM_PYTHON}" vlm_diagnose.py \
      --stage "${stage}" --work_dir "${WORK_DIR}" --shard_id "${shard}" --variant "${variant}" &
    PIDS+=("$!")
  done
  # Poll child liveness then reap each explicit PID. Unlike wait -n this also
  # handles both cached workers exiting before the parent enters its wait.
  while [[ "${#PIDS[@]}" -gt 0 ]]; do
    local remaining=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        remaining+=("${pid}")
      elif ! wait "${pid}"; then
        echo "[ERROR] ${stage}/${variant} failed. Inspect log; SAME command resumes completed candidates." >&2
        return 1
      fi
    done
    PIDS=("${remaining[@]}")
    [[ "${#PIDS[@]}" == 0 ]] || sleep 0.2
  done
}
if [[ "${RUN_REVIEW}" == 1 ]]; then
  for variant in A_original B_prompt C_pixels D_prompt_pixels; do
    echo "Paired arm ${variant}"
    run_workers review "${variant}"
  done
fi
if [[ "${RUN_SANITY}" == 1 ]]; then run_workers sanity A_original; fi
if [[ "${RUN_EVAL}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" vlm_diagnose.py --stage evaluate --work_dir "${WORK_DIR}"
fi
echo "Finished: ${WORK_DIR}/results/diagnosis.txt"
echo "Inspect input_audit/, */traces/, sanity/ and results/. NO new model checkpoint is produced."
