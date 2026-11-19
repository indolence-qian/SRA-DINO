#!/usr/bin/env bash
# CPU-only heatmaps from the completed four-dataset prediction cache.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
WORK_DIR="${WORK_DIR:-./checkpoint/residual_hr_text_four_datasets_v1}"
FLAGS=()
[[ -z "${OUTPUT_DIR:-}" ]] || FLAGS+=(--output-dir "${OUTPUT_DIR}")
[[ -z "${TRAIN_UNTIL_EPOCH:-}" ]] || FLAGS+=(--until-epoch "${TRAIN_UNTIL_EPOCH}")
for flag in SHOW_GT PACK; do
  value="${!flag:-}"
  [[ -z "${value}" || "${value}" == 0 || "${value}" == 1 ]] || { echo "${flag} must be 0 or 1" >&2; exit 1; }
done
[[ "${SHOW_GT:-1}" == 1 ]] || FLAGS+=(--no-gt)
[[ "${PACK:-0}" == 0 ]] || FLAGS+=(--pack)
"${TRAIN_PYTHON}" render_four_dataset_heatmaps.py \
  --work-dir "${WORK_DIR}" --seed "${TRAIN_SEED:-42}" \
  --datasets "${DATASETS:-mvtec,btad,mpdd,ld}" --categories "${CATEGORIES:-}" \
  --limit-per-category "${LIMIT_PER_CATEGORY:-0}" --max-side "${MAX_SIDE:-512}" \
  --normalization "${NORMALIZATION:-source}" --temperature "${TEMPERATURE:-1}" \
  --low-percentile "${LOW_PERCENTILE:-1}" --high-percentile "${HIGH_PERCENTILE:-99}" \
  --alpha "${OVERLAY_ALPHA:-0.45}" "${FLAGS[@]}" "$@"
