#!/usr/bin/env bash
# Frozen existing CLIP+DINO / MARA -> LD tests -> audited report ZIP.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
WEIGHT_PATH="${WEIGHT_PATH:-${MARA_CKPT:-${BASE_CKPT:-}}}"
[[ -n "${WEIGHT_PATH}" ]] || { echo 'Set WEIGHT_PATH to the existing checkpoint file or ckpt directory.' >&2; exit 1; }
GPU_ID="${GPU_ID:-0}"
[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || { echo 'GPU_ID must be one GPU number.' >&2; exit 1; }
WORK_DIR="${WORK_DIR:-./results/ld_$(date +'%Y%m%d_%H%M%S')}"
mkdir -p "${WORK_DIR}"
command -v flock >/dev/null || { echo 'flock is required (util-linux).' >&2; exit 1; }
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo 'Another process owns this WORK_DIR.' >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run.log") 2>&1
export CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CPU_THREADS:-4}" MKL_NUM_THREADS="${CPU_THREADS:-4}"
export LD_DATA_ROOT="${LD_DATA_ROOT:-/mnt/qfg/Tate_qfg/data/LD_AD_20261116}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXTRA=()
for flag in VERIFY_DATA SAVE_VIS CHECK_ONLY; do
  value="${!flag:-}"
  [[ -z "${value}" || "${value}" == 0 || "${value}" == 1 ]] || { echo "${flag} must be 0 or 1" >&2; exit 1; }
done
[[ "${VERIFY_DATA:-1}" == 1 ]] || EXTRA+=(--skip-disk-audit)
[[ "${SAVE_VIS:-1}" == 1 ]] || EXTRA+=(--no-vis)
[[ "${CHECK_ONLY:-0}" == 0 ]] || EXTRA+=(--check-only)
echo "LD: frozen checkpoint, GPU=${GPU_ID}, data=${LD_DATA_ROOT}, output=${WORK_DIR}"
"${TRAIN_PYTHON}" evaluate_ld.py \
  --weight-path "${WEIGHT_PATH}" --mode "${MODE:-auto}" \
  --data-root "${LD_DATA_ROOT}" --work-dir "${WORK_DIR}" \
  --archive "${DATA_ARCHIVE:-$(dirname -- "${LD_DATA_ROOT}")/LD_AD_20261116_bundle.tar.gz}" \
  --device cuda:0 --batch-size "${BATCH_SIZE:-4}" --workers "${NUM_WORKERS:-2}" \
  --image-size "${IMAGE_SIZE:-512}" --hfa-setting "${HFA_SETTING:-hfa3}" \
  --dino-repo-dir "${DINO_REPO_DIR:-./dinov3}" \
  --dino-model-name "${DINO_MODEL_NAME:-dinov3_vitl16}" \
  --dino-weights "${DINO_WEIGHTS:-./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}" \
  --clip-weights "${CLIP_WEIGHTS:-./CLIP/ckpt/ViT-L-14-336px.pt}" \
  --threshold "${THRESHOLD:-0.5}" --pro-num-th "${PRO_NUM_TH:-1000}" \
  --vis-per-category "${VIS_PER_CATEGORY:-8}" --seed "${SEED:-42}" \
  "${EXTRA[@]}" "$@"
