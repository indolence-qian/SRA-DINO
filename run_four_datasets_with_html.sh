#!/usr/bin/env bash
# Unified entry: current results first; explicit training always ends with HTML.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
[[ -z "${RUN_EXPERIMENT:-}" || "${RUN_EXPERIMENT}" == 0 || "${RUN_EXPERIMENT}" == 1 ]] || { echo 'RUN_EXPERIMENT must be 0 or 1' >&2; exit 1; }
# Keep old explicit RUN_EXPERIMENT commands working. Unset means safe auto mode.
PIPELINE_MODE="${PIPELINE_MODE:-${RUN_EXPERIMENT:+legacy}}"
case "${PIPELINE_MODE:-auto}" in
  legacy) [[ "${RUN_EXPERIMENT}" == 1 ]] && PIPELINE_MODE=train || PIPELINE_MODE=visualize ;;
  auto|visualize|resume|train) PIPELINE_MODE="${PIPELINE_MODE:-auto}" ;;
  *) echo 'PIPELINE_MODE must be auto, visualize, resume, or train' >&2; exit 1 ;;
esac
if [[ -z "${WORK_DIR:-}" ]]; then
  if [[ "${PIPELINE_MODE}" == train ]]; then
    WORK_DIR="./checkpoint/residual_hr_text_four_datasets_$(TZ=Asia/Shanghai date +'%Y%m%d_%H%M%S')"
  else
    WORK_DIR=./checkpoint/residual_hr_text_four_datasets_v1
  fi
fi
WORK_DIR="${WORK_DIR//\\//}"
export WORK_DIR
if [[ "${PIPELINE_MODE}" == auto ]]; then
  [[ -f "${WORK_DIR}/config.json" ]] || { echo "Existing experiment not found: ${WORK_DIR}. Set WORK_DIR to your completed run, or PIPELINE_MODE=train for a new experiment." >&2; exit 1; }
  if [[ -n "${TRAIN_UNTIL_EPOCH:-}" ]]; then
    PREDICTION_LOCK="${WORK_DIR}/predictions/seed${TRAIN_SEED:-42}/through${TRAIN_UNTIL_EPOCH}/locked.json"
  else
    PREDICTION_LOCK="${WORK_DIR}/predictions/seed${TRAIN_SEED:-42}/through*/locked.json"
  fi
  if compgen -G "${PREDICTION_LOCK}" >/dev/null; then
    PIPELINE_MODE=visualize
  else
    PIPELINE_MODE=resume
  fi
fi
echo "Four-dataset pipeline: mode=${PIPELINE_MODE}, experiment=${WORK_DIR}"
case "${PIPELINE_MODE}" in
  train)
    bash run_train_test_four_datasets.sh
    RESUME_PREDICTIONS=0 bash run_build_four_dataset_dashboard.sh "$@"
    ;;
  resume)
    echo 'Reuse existing trained weights; recover prediction/evaluation and build HTML.'
    RESUME_PREDICTIONS=1 bash run_build_four_dataset_dashboard.sh "$@"
    ;;
  visualize)
    echo 'Reuse latest completed predictions; generate HTML on CPU.'
    RESUME_PREDICTIONS=0 bash run_build_four_dataset_dashboard.sh "$@"
    ;;
esac
