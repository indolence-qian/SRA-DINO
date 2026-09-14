#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
MODE="${MODE:-reparse}"
if [[ "${MODE}" == reparse ]]; then
  OUT="${WORK_DIR:-./checkpoint/vlm_diagnose_visa_repaired_v2}"
  mkdir -p "${OUT}"
  command -v flock >/dev/null || { echo "Install util-linux (flock)" >&2; exit 1; }
  exec 9>"${OUT}/.run.lock"
  flock -n 9 || { echo "Output in use" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" reparse_vlm_diagnosis.py \
    --source_dir "${SOURCE_DIAG_DIR:-./checkpoint/vlm_diagnose_visa_20260912}" --output_dir "${OUT}"
elif [[ "${MODE}" == ablation ]]; then
  ROOT_OUT="${WORK_DIR:-./checkpoint/vlm_context_reference_v2}"
  # Hold detector, candidates, prompt, parser, pixel budget and calibration
  # fixed. Only observation context and reference presence change across arms.
  export LOCAL_PARSER=repair_v2 LOCAL_PROMPT=repaired INCLUDE_SUPPRESSION=0
  export LOCAL_TEACHER_MIN_PIXELS="${LOCAL_TEACHER_MIN_PIXELS:-65536}"
  export CALIBRATION_ALPHA="${CALIBRATION_ALPHA:-0.25}"
  for arm in R0_repaired R1_context R2_context_reference; do
    if [[ "${arm}" == R0_repaired ]]; then
      export LOCAL_CONTEXT_FACTOR=4 LOCAL_CONTEXT_MINIMUM=32 NORMAL_REFERENCE=0 REFERENCE_STRUCTURE_MIN=0
    else
      export LOCAL_CONTEXT_FACTOR=8 LOCAL_CONTEXT_MINIMUM=64 NORMAL_REFERENCE=0 REFERENCE_STRUCTURE_MIN=0
      if [[ "${arm}" == R2_context_reference ]]; then
        export NORMAL_REFERENCE=1 REFERENCE_STRUCTURE_MIN="${REPAIR_REFERENCE_STRUCTURE_MIN:-0.5}"
      fi
    fi
    WORK_DIR="${ROOT_OUT}/${arm}" bash run_exp_dual_vlm_local.sh
  done
else
  echo "MODE must be reparse or ablation" >&2
  exit 1
fi
