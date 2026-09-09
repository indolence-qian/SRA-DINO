#!/usr/bin/env bash
set -euo pipefail

# Frozen dual-tower experiment. No Base/MARA training and no single-tower code.
# Defaults to a source-domain pilot; DATASETS="mvtec btad mpdd" MAX_PER_CATEGORY=0
# selects full target evaluation with the SAME precommitted calibration settings.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
BASE_CKPT="${BASE_CKPT:-}"
WORK_DIR="${WORK_DIR:-./checkpoint/clip_dino_vlm_pilot}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
VLM_PYTHON="${VLM_PYTHON:-/root/miniconda3/envs/sra_vlm/bin/python}"
VLM_MODEL_ID="${VLM_MODEL_ID:-/mnt/qfg/Tate_qfg/models/Qwen3-VL-8B-Instruct-FP8}"
DATASETS="${DATASETS:-visa}"
MAX_PER_CATEGORY="${MAX_PER_CATEGORY:-40}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
RUN_EXPORT="${RUN_EXPORT:-1}"
RUN_REVIEW="${RUN_REVIEW:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
VLM_HEATMAP="${VLM_HEATMAP:-0}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ -z "${BASE_CKPT}" || ! -e "${BASE_CKPT}" ]]; then
  echo "[ERROR] BASE_CKPT must be the existing dual-tower stage-one .pth file or ckpt directory." >&2
  exit 1
fi
if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "[ERROR] GPU_IDS must be comma-separated physical GPU indices." >&2
  exit 1
fi
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "[ERROR] Duplicate GPU id: ${gpu}" >&2
    exit 1
  fi
  SEEN_GPUS[${gpu}]=1
done
if [[ -n "${NPROC_PER_NODE:-}" && "${NPROC_PER_NODE}" != "${#GPUS[@]}" ]]; then
  echo "[ERROR] NPROC_PER_NODE must match GPU_IDS; these workers use sharding, not DDP." >&2
  exit 1
fi
for flag in "${RUN_EXPORT}" "${RUN_REVIEW}" "${RUN_EVAL}" "${VLM_HEATMAP}"; do
  [[ "${flag}" == 0 || "${flag}" == 1 ]] || { echo "[ERROR] Stage flags must be 0 or 1." >&2; exit 1; }
done
mkdir -p "${WORK_DIR}"
if ! command -v flock >/dev/null 2>&1; then
  echo "[ERROR] Install util-linux (flock) to protect resumable work directories." >&2
  exit 1
fi
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo "[ERROR] Another run is using ${WORK_DIR}." >&2; exit 1; }
LOG_FILE="${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG_FILE}") 2>&1
export PYTHONUNBUFFERED=1
echo "Frozen CLIP+DINO -> ROI review -> fixed calibration evaluation (NO retraining/MARA)"
echo "DATASETS=${DATASETS}, max/category=${MAX_PER_CATEGORY}, GPU_IDS=${GPU_IDS}"
echo "WORK_DIR=${WORK_DIR}, log=${LOG_FILE}"

# Validate both environments before spending time exporting the dataset.
if [[ "${RUN_EXPORT}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == ${#GPUS[@]}; print('Export torch:', torch.__version__)"
fi
if [[ "${RUN_REVIEW}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${VLM_PYTHON}" -c "import os,shutil,torch,vllm,qwen_vl_utils; from triton.runtime import driver; assert torch.cuda.is_available() and torch.cuda.device_count() == ${#GPUS[@]}; assert shutil.which(os.environ.get('CC','gcc')) or (not os.environ.get('CC') and shutil.which('clang')), 'Install build-essential or set CC to an existing compiler'; print('VLM torch/vLLM:', torch.__version__,vllm.__version__); print('Triton CUDA:',driver.active.get_current_device())"
fi
HEATMAP_FLAG=()
[[ "${VLM_HEATMAP}" == 0 ]] || HEATMAP_FLAG=(--heatmap)
"${TRAIN_PYTHON}" dual_vlm.py --stage prepare \
  --work_dir "${WORK_DIR}" --base_ckpt "${BASE_CKPT}" \
  --datasets "${DATASETS}" --max_per_category "${MAX_PER_CATEGORY}" \
  --num_shards "${#GPUS[@]}" --seed "${SEED:-42}" \
  --batch_size "${EXPORT_BS:-4}" --image_size "${IMAGE_SIZE:-512}" \
  --roi_fraction "${ROI_FRACTION:-0.25}" \
  --model_id "${VLM_MODEL_ID}" --teacher_image_size "${VLM_IMAGE_SIZE:-512}" \
  --gpu_memory "${VLM_GPU_MEMORY:-0.70}" --max_model_len "${VLM_MAX_MODEL_LEN:-4096}" \
  --max_tokens "${VLM_MAX_TOKENS:-512}" --retries "${VLM_RETRIES:-1}" \
  --max_invalid_ratio "${VLM_MAX_INVALID_RATIO:-0.05}" \
  --alpha "${CALIBRATION_ALPHA:-0.5}" --confidence_threshold "${VLM_CONFIDENCE_THRESHOLD:-0.8}" \
  --pro_num_th "${PRO_NUM_TH:-1000}" --pro_max_fpr "${PRO_MAX_FPR:-0.3}" \
  --hfa_setting "${HFA_SETTING:-hfa3}" --dino_bottleneck "${DINO_BOTTLENECK:-256}" \
  --dino_repo_dir "${DINO_REPO_DIR:-./dinov3}" --dino_model_name "${DINO_MODEL_NAME:-dinov3_vitl16}" \
  --dino_weights "${DINO_WEIGHTS:-./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}" \
  --clip_model_name "${CLIP_MODEL_NAME:-ViT-L-14-336}" --clip_pretrained "${CLIP_PRETRAINED:-openai}" \
  --visual_layers "${VISUAL_LAYERS:-5,11,17,23}" "${HEATMAP_FLAG[@]}"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_workers() {
  local stage="$1" runtime="$2"
  PIDS=()
  for shard in "${!GPUS[@]}"; do
    echo "Starting ${stage} shard ${shard} on physical GPU ${GPUS[${shard}]}"
    CUDA_VISIBLE_DEVICES="${GPUS[${shard}]}" "${runtime}" dual_vlm.py \
      --stage "${stage}" --work_dir "${WORK_DIR}" --shard_id "${shard}" &
    PIDS+=("$!")
  done
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[${index}]}"; then
      unset 'PIDS[index]'
      echo "[ERROR] ${stage} worker failed; rerun SAME command to resume."
      return 1
    fi
    unset 'PIDS[index]'
  done
  PIDS=()
}

if [[ "${RUN_EXPORT}" == 1 ]]; then
  echo "===== Stage A: frozen dual-tower evidence export ====="
  run_workers export "${TRAIN_PYTHON}"
  "${TRAIN_PYTHON}" dual_vlm.py --stage seal --work_dir "${WORK_DIR}"
fi
if [[ "${RUN_REVIEW}" == 1 ]]; then
  echo "===== Stage B: dual-GPU FP8 ROI review; dual tower is now unloaded ====="
  run_workers review "${VLM_PYTHON}"
fi
if [[ "${RUN_EVAL}" == 1 ]]; then
  echo "===== Stage C: CPU calibration/metrics from cached predictions ====="
  CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" dual_vlm.py --stage evaluate --work_dir "${WORK_DIR}"
fi
echo "Finished. Results: ${WORK_DIR}/results/metric_vlm.txt"
