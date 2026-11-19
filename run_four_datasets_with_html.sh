#!/usr/bin/env bash
# Recommended future entry: unchanged training/evaluation + automatic HTML report.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export WORK_DIR="${WORK_DIR:-./checkpoint/residual_hr_text_four_datasets_v1}"
[[ "${RUN_EXPERIMENT:-1}" == 0 || "${RUN_EXPERIMENT:-1}" == 1 ]] || { echo 'RUN_EXPERIMENT must be 0 or 1' >&2; exit 1; }
if [[ "${RUN_EXPERIMENT:-1}" == 1 ]]; then
  bash run_train_test_four_datasets.sh
fi
bash run_build_four_dataset_dashboard.sh "$@"
