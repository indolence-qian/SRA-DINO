#!/usr/bin/env bash
set -euo pipefail
# One-command frozen-Base export -> single-candidate native VLM -> P0/P1 audit.
# All parameters may be supplied inline with nohup env. No training starts here.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LOCAL_REVIEW=1
export WORK_DIR="${WORK_DIR:-./checkpoint/dual_vlm_local_visa_v1}"
export CALIBRATION_ALPHA="${CALIBRATION_ALPHA:-0.25}"
export NORMAL_REFERENCE="${NORMAL_REFERENCE:-0}"
export INCLUDE_SUPPRESSION="${INCLUDE_SUPPRESSION:-0}"
for flag in "${NORMAL_REFERENCE}" "${INCLUDE_SUPPRESSION}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo "[ERROR] Reference/suppression flags must be 0 or 1" >&2; exit 1; }
done
exec bash "${SCRIPT_DIR}/run_exp_dual_vlm.sh"
