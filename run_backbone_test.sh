#!/usr/bin/env bash
set -euo pipefail

# =========================
# Basic config
# =========================
PYTHON_BIN="python"
DEVICE="cuda:0"
VISUAL_BACKBONE="clip"          # dino / clip
METHOD_NAME="CLIP-Visual"       # used in exported paper tables
WEIGHT_PATH="./checkpoint/learnablePrompt/20260406_200500/ckpt"
SAVE_ROOT="./TESTING_ALL/clip"
DATASETS=("mvtec" "btad" "mpdd")
CATEGORY="ALL"
BATCH_SIZE=32
IMAGE_SIZE=512
NUM_WORKERS=0

# Optional: run training before testing
RUN_TRAIN_FIRST=0                # 1: run training command first; 0: skip training
TRAIN_CMD=""                    # Example: python train_up.py --result_path ./checkpoint/exp --device cuda:0 --dataset visa
TRAIN_WORKDIR="."
# If training writes ckpts under result_path/ckpt, set this to 1 and WEIGHT_PATH will be updated automatically.
USE_TRAIN_RESULT_AS_WEIGHT_PATH=0
TRAIN_RESULT_PATH="./checkpoint/exp"

# CLIP / DINO config
CLIP_MODEL_NAME="ViT-L-14-336"
CLIP_PRETRAINED="openai"
DINO_REPO_DIR="./dinov3"
DINO_MODEL_NAME="dinov3_vitl16"
DINO_WEIGHTS="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DINO_BOTTLENECK=256
VISUAL_LAYERS=(5 11 17 23)

# Eval config
NORM_MODE="none"
PRO_NUM_TH=1000
PRO_MAX_FPR=0.3
EVAL_LATEST_ONLY=0               # 1: only latest ckpt, 0: all ckpts and auto-select best epoch
SAVE_VIS=1                       # 1: save overlay figures, 0: metrics only
SELECTION_METRIC="mean_PRO"     # mean_F1 / mean_I_AUROC / mean_P_AUROC / mean_PRO
EXPORT_PAPER_TABLES=1            # 1: export per-dataset LaTeX tables

# Logging
LOG_DIR="${SAVE_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/pipeline_${VISUAL_BACKBONE}_${TS}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date +'%F %T')] Pipeline start"
echo "LOG_FILE=${LOG_FILE}"
echo "VISUAL_BACKBONE=${VISUAL_BACKBONE}"
echo "METHOD_NAME=${METHOD_NAME}"
echo "WEIGHT_PATH=${WEIGHT_PATH}"

if [[ "${RUN_TRAIN_FIRST}" == "1" ]]; then
  if [[ -z "${TRAIN_CMD}" ]]; then
    echo "[ERROR] RUN_TRAIN_FIRST=1 but TRAIN_CMD is empty."
    exit 1
  fi
  echo "===== Start Training ====="
  echo "TRAIN_CMD=${TRAIN_CMD}"
  (cd "${TRAIN_WORKDIR}" && eval "${TRAIN_CMD}")
  echo "===== Training Finished ====="

  if [[ "${USE_TRAIN_RESULT_AS_WEIGHT_PATH}" == "1" ]]; then
    WEIGHT_PATH="${TRAIN_RESULT_PATH}/ckpt"
    echo "[INFO] Updated WEIGHT_PATH=${WEIGHT_PATH}"
  fi
fi

COMMON_ARGS=(
  --weight_path "${WEIGHT_PATH}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --category "${CATEGORY}"
  --image_size "${IMAGE_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --visual_backbone "${VISUAL_BACKBONE}"
  --clip_model_name "${CLIP_MODEL_NAME}"
  --clip_pretrained "${CLIP_PRETRAINED}"
  --dino_repo_dir "${DINO_REPO_DIR}"
  --dino_model_name "${DINO_MODEL_NAME}"
  --dino_weights "${DINO_WEIGHTS}"
  --dino_bottleneck "${DINO_BOTTLENECK}"
  --norm_mode "${NORM_MODE}"
  --pro_num_th "${PRO_NUM_TH}"
  --pro_max_fpr "${PRO_MAX_FPR}"
  --selection_metric "${SELECTION_METRIC}"
  --paper_method_name "${METHOD_NAME}"
  --visual_layers "${VISUAL_LAYERS[@]}"
)

if [[ "${EVAL_LATEST_ONLY}" == "1" ]]; then
  COMMON_ARGS+=(--eval_latest_only)
fi
if [[ "${SAVE_VIS}" == "1" ]]; then
  COMMON_ARGS+=(--save_vis)
fi
if [[ "${EXPORT_PAPER_TABLES}" == "1" ]]; then
  COMMON_ARGS+=(--export_paper_tables)
fi

mkdir -p "${SAVE_ROOT}"
for ds in "${DATASETS[@]}"; do
  echo "===== Start testing on ${ds} ====="
  "${PYTHON_BIN}" test_backbone_eval.py \
    --result_path "${SAVE_ROOT}" \
    --dataset "${ds}" \
    "${COMMON_ARGS[@]}"
  echo "===== Finished ${ds} ====="
done

echo "===== Aggregating best epochs across datasets ====="
METHOD_NAME_ENV="${METHOD_NAME}" SAVE_ROOT_ENV="${SAVE_ROOT}" SELECTION_METRIC_ENV="${SELECTION_METRIC}" PYTHONHASHSEED=0 "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

save_root = Path(os.environ["SAVE_ROOT_ENV"])
method_name = os.environ["METHOD_NAME_ENV"]
selection_metric = os.environ["SELECTION_METRIC_ENV"]
paper_dir = save_root / "paper_tables"
paper_dir.mkdir(parents=True, exist_ok=True)

rows = []
for dataset_dir in sorted([p for p in save_root.iterdir() if p.is_dir() and p.name not in {"logs", "paper_tables"}]):
    best_json = dataset_dir / "best_epoch.json"
    if not best_json.exists():
        continue
    data = json.loads(best_json.read_text(encoding="utf-8"))
    rows.append({
        "dataset": data["dataset"],
        "method": method_name,
        "epoch_name": data["epoch_name"],
        "selection_metric": data["selection_metric"],
        "mean_F1": float(data["mean_F1"]),
        "mean_I_AUROC": float(data["mean_I_AUROC"]),
        "mean_P_AUROC": float(data["mean_P_AUROC"]),
        "mean_PRO": float(data["mean_PRO"]),
        "ckpt_path": data.get("ckpt_path", ""),
    })

if not rows:
    raise SystemExit("No best_epoch.json files found under SAVE_ROOT.")

csv_path = paper_dir / "best_by_dataset.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["dataset", "method", "epoch_name", "selection_metric", "mean_F1", "mean_I_AUROC", "mean_P_AUROC", "mean_PRO", "ckpt_path"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

avg = {
    "mean_F1": sum(r["mean_F1"] for r in rows) / len(rows),
    "mean_I_AUROC": sum(r["mean_I_AUROC"] for r in rows) / len(rows),
    "mean_P_AUROC": sum(r["mean_P_AUROC"] for r in rows) / len(rows),
    "mean_PRO": sum(r["mean_PRO"] for r in rows) / len(rows),
}

latex_lines = [
    "\\begin{table}[t]",
    "\\centering",
    f"\\caption{{Best results selected by {selection_metric} for {method_name} across datasets.}}",
    f"\\label{{tab:{method_name.lower().replace(' ', '_').replace('-', '_')}_best_by_dataset}}",
    "\\resizebox{0.98\\linewidth}{!}{%",
    "\\begin{tabular}{lccccc}",
    "\\toprule",
    "Dataset & Method & F1 (\\%) & I-AUROC (\\%) & P-AUROC (\\%) & PRO (\\%) \\",
    "\\midrule",
]
for row in rows:
    latex_lines.append(
        f"{row['dataset']} & {row['method']} & {row['mean_F1'] * 100:.2f} & {row['mean_I_AUROC'] * 100:.2f} & {row['mean_P_AUROC'] * 100:.2f} & {row['mean_PRO'] * 100:.2f} \\")
latex_lines += [
    "\\midrule",
    f"Average & {method_name} & {avg['mean_F1'] * 100:.2f} & {avg['mean_I_AUROC'] * 100:.2f} & {avg['mean_P_AUROC'] * 100:.2f} & {avg['mean_PRO'] * 100:.2f} \\",
    "\\bottomrule",
    "\\end{tabular}}",
    "\\end{table}",
    "",
]
(paper_dir / "best_by_dataset.tex").write_text("\n".join(latex_lines), encoding="utf-8")

summary_lines = [
    f"Method: {method_name}",
    f"Selection metric: {selection_metric}",
    f"Datasets: {', '.join(r['dataset'] for r in rows)}",
    "",
]
for row in rows:
    summary_lines += [
        f"[{row['dataset']}] best epoch: {row['epoch_name']}",
        f"  F1={row['mean_F1']:.6f}, I-AUROC={row['mean_I_AUROC']:.6f}, P-AUROC={row['mean_P_AUROC']:.6f}, PRO={row['mean_PRO']:.6f}",
        f"  ckpt={row['ckpt_path']}",
        "",
    ]
summary_lines.append(
    f"Average: F1={avg['mean_F1']:.6f}, I-AUROC={avg['mean_I_AUROC']:.6f}, P-AUROC={avg['mean_P_AUROC']:.6f}, PRO={avg['mean_PRO']:.6f}"
)
(paper_dir / "best_by_dataset.txt").write_text("\n".join(summary_lines), encoding="utf-8")

print(f"[AGG] Wrote {csv_path}")
print(f"[AGG] Wrote {paper_dir / 'best_by_dataset.tex'}")
print(f"[AGG] Wrote {paper_dir / 'best_by_dataset.txt'}")
PY

echo "[$(date +'%F %T')] Pipeline done"
