#!/usr/bin/env bash
# Read-only reuse of the completed frozen lambda run; no training or inference.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
LAMBDA_DIR="${LAMBDA_DIR:-./checkpoint/text_lambda_v1_through15}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoint/fixed_t3_lambda2_$(TZ=Asia/Shanghai date +'%Y%m%d_%H%M%S')}"
command -v flock >/dev/null || { echo 'flock is required' >&2; exit 1; }
exec 8<"${LAMBDA_DIR}/.run.lock"
flock -sn 8 || { echo 'Lambda experiment is still running' >&2; exit 1; }
"${TRAIN_PYTHON:-python}" fixed_text_configuration.py \
  --lambda-dir "${LAMBDA_DIR}" --output-dir "${OUTPUT_DIR}" \
  --sample-fraction "${SAMPLE_FRACTION:-0.1}"
