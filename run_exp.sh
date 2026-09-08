#!/usr/bin/env bash
set -euo pipefail

# Mainline: original CLIP + DINO detector -> MARA -> cross-dataset tests.
# The single-tower/VLM experiment is retained as an explicitly selected ablation.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
BASE_ARCH="${BASE_ARCH:-clip_dino}"

case "${BASE_ARCH}" in
  dino_single)
    exec bash "${SCRIPT_DIR}/run_exp_dino_single.sh" "$@"
    ;;
  clip_dino) ;;
  *)
    echo "[ERROR] BASE_ARCH must be clip_dino or dino_single, got: ${BASE_ARCH}" >&2
    exit 1
    ;;
esac

# Do not silently reinterpret an old single-tower VLM command as a dual tower.
if [[ "${RUN_VLM_CACHE:-0}" != "0" || "${RUN_VLM_DISTILL:-0}" != "0" ]]; then
  echo "[ERROR] VLM caching/distillation currently supports only dino_single, not clip_dino." >&2
  echo "[ERROR] For the restored dual tower, set RUN_VLM_CACHE=0 RUN_VLM_DISTILL=0." >&2
  echo "[ERROR] To reproduce the single-tower ablation, set BASE_ARCH=dino_single." >&2
  exit 1
fi
if [[ "${RUN_TEST:-1}" == "1" && "${RUN_MARA:-1}" != "1" ]]; then
  echo "[ERROR] Integrated RUN_TEST=1 requires RUN_MARA=1. For existing MARA checkpoints use test_mara_final.sh." >&2
  exit 1
fi

export DATASET="${SOURCE_DATASET:-${DATASET:-visa}}"
export HFA_SETTING="${HFA_SETTING:-hfa3}"
export GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export DEVICE="${DEVICE:-cuda:0}"
export TRICK_NAME="${TRICK_NAME:-clip_dino_mara}"
export TEST_TRICK_NAME="${TEST_TRICK_NAME:-${TRICK_NAME}_final}"
# Never choose the best checkpoint using unseen target test metrics by default.
export TEST_EVAL_LATEST_ONLY="${TEST_EVAL_LATEST_ONLY:-1}"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || (( ${#GPU_LIST[@]} != NPROC_PER_NODE )); then
  echo "[ERROR] GPU_IDS must contain NPROC_PER_NODE GPUs (got ${GPU_IDS}, ${NPROC_PER_NODE})." >&2
  exit 1
fi

if [[ "${RUN_BASE:-1}" != "1" ]]; then
  # Inspect metadata rather than filenames. Legacy dual checkpoints omit base_arch.
  python - "${BASE_CKPT:-}" <<'PY'
import os
import sys
import torch

path = sys.argv[1]
if not path or not os.path.isfile(path):
    raise SystemExit("[ERROR] RUN_BASE=0 requires an existing dual-tower BASE_CKPT.")
payload = torch.load(path, map_location="cpu", weights_only=False)
arch = payload.get("base_arch", "clip_dino")
if arch != "clip_dino" or "dino_single_config" in payload:
    raise SystemExit(f"[ERROR] Expected a CLIP+DINO checkpoint, got {arch!r}: {path}")
print(f"Verified dual-tower checkpoint: {path}")
PY
fi

echo "Restored mainline: BASE_ARCH=clip_dino, CLIP + DINO, VLM disabled."
echo "Stage 1: original single-GPU trainer (${DEVICE} within GPUs=${GPU_IDS}); MARA: ${NPROC_PER_NODE}-GPU DDP; tests: dataset-parallel."
exec bash "${SCRIPT_DIR}/train_mara_visa.sh" "$@"
