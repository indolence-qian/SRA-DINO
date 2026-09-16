#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
SOURCE_WORK_DIR="${SOURCE_WORK_DIR:-./checkpoint/vlm_context_reference_v2/R2_context_reference}"
WORK_DIR="${WORK_DIR:-./checkpoint/vlm_next_reference_action_v1}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
VLM_PYTHON="${VLM_PYTHON:-/root/miniconda3/envs/sra_vlm/bin/python}"
GPU_IDS="${GPU_IDS:-0,1}"
RUN_REVIEW="${RUN_REVIEW:-1}"
RUN_ACTION="${RUN_ACTION:-1}"
RUN_REFERENCE_EVAL="${RUN_REFERENCE_EVAL:-1}"
for flag in "${RUN_REVIEW}" "${RUN_ACTION}" "${RUN_REFERENCE_EVAL}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo "Stage flags must be 0/1" >&2; exit 1; }
done
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid GPU_IDS" >&2; exit 1; }
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[$gpu]:-}" ]] || { echo "Duplicate GPU id" >&2; exit 1; }
  SEEN[$gpu]=1
done
[[ -f "${SOURCE_WORK_DIR}/manifest.json" ]] || { echo "SOURCE_WORK_DIR must contain the FULL R2 export, not just CSVs" >&2; exit 1; }
command -v "${TRAIN_PYTHON}" >/dev/null || { echo "Missing TRAIN_PYTHON=${TRAIN_PYTHON}" >&2; exit 1; }
if [[ "${RUN_REVIEW}" == 1 ]]; then
  command -v "${VLM_PYTHON}" >/dev/null || { echo "Missing VLM_PYTHON=${VLM_PYTHON}; pass the actual sra_vlm python path" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${VLM_PYTHON}" -c "import os,shutil,torch,vllm,qwen_vl_utils; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; assert shutil.which(os.environ.get('CC','gcc')) or shutil.which('clang'), 'Install build-essential'; print('VLM preflight:',torch.__version__,vllm.__version__)"
fi
"${TRAIN_PYTHON}" -c "import numpy,cv2,sklearn,PIL; print('CPU diagnostics dependencies ready')"
"${TRAIN_PYTHON}" -c "from pathlib import Path; import sys; s,o=map(lambda p:Path(p).resolve(),sys.argv[1:]); assert s!=o and not s.is_relative_to(o) and not o.is_relative_to(s), 'SOURCE_WORK_DIR and WORK_DIR must not overlap'" "${SOURCE_WORK_DIR}" "${WORK_DIR}"
command -v flock >/dev/null || { echo "Install util-linux (flock)" >&2; exit 1; }
mkdir -p "${WORK_DIR}"
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo "WORK_DIR already in use" >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
export PYTHONUNBUFFERED=1
echo "Post-R2 diagnostics: frozen export reuse, NO training, NO source overwrite"
MODEL_FLAG=()
[[ -z "${VLM_MODEL_ID:-}" ]] || MODEL_FLAG=(--model_id "${VLM_MODEL_ID}")
CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" vlm_next_diagnose.py --stage prepare \
  --source_dir "${SOURCE_WORK_DIR}" --work_dir "${WORK_DIR}" \
  --candidates "${AUDIT_CANDIDATES:-192}" --seed "${SEED:-42}" --num_shards "${#GPUS[@]}" \
  --min_pixels "${LOCAL_TEACHER_MIN_PIXELS:-65536}" --alphas "${AUDIT_ALPHAS:-0.25}" \
  --gpu_memory "${VLM_GPU_MEMORY:-0.70}" --max_tokens "${VLM_MAX_TOKENS:-768}" \
  --max_model_len "${VLM_MAX_MODEL_LEN:-4096}" --retries "${VLM_RETRIES:-1}" \
  --max_invalid_ratio "${VLM_MAX_INVALID_RATIO:-0.05}" "${MODEL_FLAG[@]}"
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "${RUN_ACTION}" == 1 ]]; then
  echo "P1: ALL source images, cached VLM vs GT diagnostic choices vs all candidates; CPU only"
  CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" vlm_next_diagnose.py --stage action_eval --work_dir "${WORK_DIR}"
fi
if [[ "${RUN_REVIEW}" == 1 ]]; then
  echo "P0: paired no-reference / retrieved-reference / shuffled-reference; independent GPU workers"
  for shard in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "${VLM_PYTHON}" vlm_next_diagnose.py \
      --stage review --work_dir "${WORK_DIR}" --shard_id "${shard}" &
    PIDS+=("$!")
  done
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$index]}"; then
      unset 'PIDS[index]'
      echo "Review failed; rerun same command to resume valid caches" >&2
      exit 1
    fi
    unset 'PIDS[index]'
  done
fi
if [[ "${RUN_REFERENCE_EVAL}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" vlm_next_diagnose.py --stage reference_eval --work_dir "${WORK_DIR}"
fi
echo "Done: ${WORK_DIR}/action_results and ${WORK_DIR}/reference_results"
