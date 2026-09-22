#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
REPAIR_WORK_DIR="${REPAIR_WORK_DIR:-./checkpoint/upstream_P0_P1_v1}"
WORK_DIR="${WORK_DIR:-./checkpoint/upstream_semantic_audit_v1}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU_IDS' >&2; exit 1; }
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[${gpu}]:-}" ]] || { echo 'Duplicate GPU ID' >&2; exit 1; }
  SEEN[${gpu}]=1
done
[[ -f "${REPAIR_WORK_DIR}/repair_config.json" ]] || { echo 'Missing P0/P1 repair_config.json' >&2; exit 1; }
# Validate paths before creating the output directory or its lock/log files.
"${TRAIN_PYTHON}" -c 'import sys,json; from pathlib import Path; from upstream_semantic_audit import non_nested; r=Path(sys.argv[1]).resolve(); s=json.loads((r/"repair_config.json").read_text())["source"]; non_nested([sys.argv[2],r,s])' "${REPAIR_WORK_DIR}" "${WORK_DIR}"
command -v flock >/dev/null || { echo 'Install util-linux (flock)' >&2; exit 1; }
mkdir -p "${WORK_DIR}"
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo 'Output directory is already in use' >&2; exit 1; }
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c "import torch,cv2,sklearn,matplotlib; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}; print(torch.__version__)"
"${TRAIN_PYTHON}" upstream_semantic_audit.py --stage prepare --work_dir "${WORK_DIR}" \
  --repair_dir "${REPAIR_WORK_DIR}" --num_shards "${#GPUS[@]}" \
  --batch_size "${AUDIT_BATCH_SIZE:-4}" --checkpoints "${AUDIT_CHECKPOINTS:-best,last}" \
  --partitions "${AUDIT_PARTITIONS:-val,eval}" --visuals_per_category "${VISUALS_PER_CATEGORY:-3}" \
  --near_radius "${NEAR_BG_RADIUS:-8}" --pixel_epsilon "${PIXEL_EPSILON:-0.000001}" \
  --residual_epsilon "${RESIDUAL_EPSILON:-0.000001}"
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
echo 'Auditing frozen P0/P1 heads. Reuses sealed features; no Qwen, training or alpha search.'
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for shard in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[${shard}]}" "${TRAIN_PYTHON}" upstream_semantic_audit.py \
    --stage evaluate --work_dir "${WORK_DIR}" --shard_id "${shard}" --workers "${LOADER_WORKERS:-2}" &
  PIDS+=("$!")
done
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[${index}]}"; then
    unset 'PIDS[index]'
    echo 'Audit worker failed; report was not generated. Resume with the same command.' >&2
    exit 1
  fi
  unset 'PIDS[index]'
done
PIDS=()
"${TRAIN_PYTHON}" upstream_semantic_audit.py --stage report --work_dir "${WORK_DIR}"
echo "Done: ${WORK_DIR}/README.md, participation.csv, metrics.csv, components.csv, regions.csv, panels/"
