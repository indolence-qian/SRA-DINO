#!/usr/bin/env bash
# Existing experiment -> interactive offline analysis; no training by default.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export WORK_DIR="${WORK_DIR:-./checkpoint/residual_hr_text_four_datasets_v1}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
[[ "${RESUME_PREDICTIONS:-0}" == 0 || "${RESUME_PREDICTIONS:-0}" == 1 ]] || { echo 'RESUME_PREDICTIONS must be 0 or 1' >&2; exit 1; }
if [[ "${RESUME_PREDICTIONS:-0}" == 1 ]]; then
  RUN_TRAIN=0 RUN_PREDICT=1 RUN_EVAL=1 bash run_train_test_four_datasets.sh
fi
FLAGS=()
[[ -z "${OUTPUT_DIR:-}" ]] || FLAGS+=(--output-dir "${OUTPUT_DIR}")
[[ -z "${TRAIN_UNTIL_EPOCH:-}" ]] || FLAGS+=(--until-epoch "${TRAIN_UNTIL_EPOCH}")
"${TRAIN_PYTHON}" build_four_dataset_dashboard.py \
  --work-dir "${WORK_DIR}" --seed "${TRAIN_SEED:-42}" \
  --datasets "${DATASETS:-mvtec,btad,mpdd,ld}" --categories "${CATEGORIES:-}" \
  --sample-fraction "${SAMPLE_FRACTION:-0.1}" \
  --limit-per-category "${LIMIT_PER_CATEGORY:-0}" --max-side "${MAX_SIDE:-512}" \
  --normalization "${NORMALIZATION:-source}" --temperature "${TEMPERATURE:-1}" \
  --low-percentile "${LOW_PERCENTILE:-1}" --high-percentile "${HIGH_PERCENTILE:-99}" \
  --alpha "${OVERLAY_ALPHA:-0.45}" "${FLAGS[@]}" "$@"
