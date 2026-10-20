#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
MARA_CKPT="${MARA_CKPT:-}"
WORK_DIR="${WORK_DIR:-./checkpoint/mara_ppo_v1}"
SOURCE_WORK_DIR="${SOURCE_WORK_DIR:-}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS//[[:space:]]/}"
CPU_THREADS="${CPU_THREADS:-4}"
DATA_ROOTS="${DATA_ROOTS:-}"
[[ -n "${DATA_ROOTS}" ]] || DATA_ROOTS='{}'
[[ -f "${MARA_CKPT}" ]] || { echo "[ERROR] MARA_CKPT must be the original TRAINED, complete MARA .pth FILE" >&2; exit 1; }
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid GPU_IDS" >&2; exit 1; }
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ -z "${SEEN[$gpu]:-}" ]] || { echo "Duplicate GPU id ${gpu}" >&2; exit 1; }
  SEEN[$gpu]=1
done
[[ "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid CPU_THREADS" >&2; exit 1; }
command -v "${TRAIN_PYTHON}" >/dev/null
command -v flock >/dev/null
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}"
"${TRAIN_PYTHON}" -c "import torch,numpy,cv2,PIL,matplotlib,train_mara,mara_ppo_experiment; print('PPO dependencies:',torch.__version__)"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_PYTHON}" -c \
  "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==${#GPUS[@]}, 'Requested GPUs unavailable'; print([(torch.cuda.get_device_name(i),round(torch.cuda.get_device_properties(i).total_memory/2**30,1)) for i in range(torch.cuda.device_count())])"
[[ -d "${WORK_DIR}" ]] || mkdir -p "${WORK_DIR}"
exec 9>"${WORK_DIR}/.run.lock"
flock -n 9 || { echo "WORK_DIR is already running" >&2; exit 1; }
exec > >(tee -a "${WORK_DIR}/run_$(date +'%Y%m%d_%H%M%S').log") 2>&1
EXTRA=()
[[ -z "${SOURCE_WORK_DIR}" ]] || EXTRA+=(--source_work_dir "${SOURCE_WORK_DIR}")
[[ -z "${DINO_REPO_DIR:-}" ]] || EXTRA+=(--dino_repo_dir "${DINO_REPO_DIR}")
[[ -z "${DINO_WEIGHTS:-}" ]] || EXTRA+=(--dino_weights "${DINO_WEIGHTS}")
[[ -z "${CATEGORIES:-}" ]] || EXTRA+=(--categories "${CATEGORIES}")
echo "PPO pilot: V vs MC; frozen trained Refiner/Base; same HR reward/split/init; no gain veto/no VLM"
if [[ "${RESUME_PREPARED:-0}" == 1 ]]; then
  # Explicitly keep the OLD sealed scientific config after a logging-only code
  # update. New scientific env overrides are ignored in this mode, not resealed.
  "${TRAIN_PYTHON}" -c "import sys,mara_ppo_experiment as e; r,c,m,s=e.load_work(sys.argv[1]); assert c['num_shards']==int(sys.argv[2]), 'GPU shard layout changed'; print('RESUME_PREPARED: using existing sealed scientific config; scientific env overrides ignored; signature='+s)" "${WORK_DIR}" "${#GPUS[@]}"
else
CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" mara_ppo_experiment.py --stage prepare \
  --work_dir "${WORK_DIR}" --mara_ckpt "${MARA_CKPT}" \
  --datasets "${DATASETS:-visa mvtec btad mpdd}" --data_roots "${DATA_ROOTS}" \
  --source_dataset "${SOURCE_DATASET:-visa}" --source_samples_per_category "${SOURCE_SAMPLES_PER_CATEGORY:-0}" \
  --samples_per_category "${SAMPLES_PER_CATEGORY:-8}" --seed "${SEED:-42}" \
  --source_split_seed "${SOURCE_SPLIT_SEED:-20261018}" --calibration_fraction "${CALIBRATION_FRACTION:-0.2}" \
  --train_seeds "${TRAIN_SEEDS:-0}" --epochs "${EPOCHS:-30}" --steps "${STEPS:-3}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-8}" --trajectories "${TRAJECTORIES:-4}" \
  --ppo_epochs "${PPO_EPOCHS:-4}" --minibatch_size "${MINIBATCH_SIZE:-16}" \
  --actor_lr "${ACTOR_LR:-0.0001}" --critic_lr "${CRITIC_LR:-0.0003}" \
  --value_probe_trajectories "${VALUE_PROBE_TRAJECTORIES:-32}" --num_shards "${#GPUS[@]}" \
  --export_batch_size "${EXPORT_BATCH_SIZE:-8}" --eval_batch_size "${EVAL_BATCH_SIZE:-8}" \
  --decode_workers "${DECODE_WORKERS:-2}" --metric_workers "${METRIC_WORKERS:-2}" --cpu_threads "${CPU_THREADS}" \
  --cpu_cache_gb "${CPU_CACHE_GB:-4}" --gpu_cache_gb "${GPU_CACHE_GB:-12}" \
  --image_size "${IMAGE_SIZE:-512}" --pro_num_th "${PRO_NUM_TH:-1000}" --pro_max_fpr "${PRO_MAX_FPR:-0.3}" \
  --text_source "${TEXT_SOURCE:-prompt_learner}" --clip_model_name "${CLIP_MODEL_NAME:-ViT-L-14-336}" \
  --clip_pretrained "${CLIP_PRETRAINED:-openai}" "${EXTRA[@]}"
fi
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_workers() {
  local stage="$1"
  PIDS=()
  for shard in "${!GPUS[@]}"; do
    local log="${WORK_DIR}/${stage}_shard_${shard}.log"
    local overrides=()
    if [[ "${stage}" == train && -n "${EXEC_GPU_CACHE_GB:-}" ]]; then
      overrides+=(--gpu_cache_gb "${EXEC_GPU_CACHE_GB}")
    fi
    if [[ "${stage}" == train ]]; then
      overrides+=(--monitor_every "${MONITOR_EVERY:-5}" --monitor_samples_per_category "${MONITOR_SAMPLES_PER_CATEGORY:-8}" --preview_samples "${PREVIEW_SAMPLES:-8}")
    fi
    echo "[$(date +'%F %T')] ${stage} GPU=${GPUS[$shard]} shard=${shard}" >> "${log}"
    echo "Launching ${stage} GPU=${GPUS[$shard]}; log=${log}"
    CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "${TRAIN_PYTHON}" mara_ppo_experiment.py \
      --stage "${stage}" --work_dir "${WORK_DIR}" --shard_id "${shard}" --device cuda:0 \
      --cpu_threads "${CPU_THREADS}" "${overrides[@]}" >> "${log}" 2>&1 &
    PIDS+=("$!")
  done
  # Fail immediately when EITHER GPU fails; never leave the other worker running
  # for hours before discovering a failed peer or collecting a partial report.
  while ((${#PIDS[@]})); do
    for index in "${!PIDS[@]}"; do
      if ! kill -0 "${PIDS[$index]}" 2>/dev/null; then
        if ! wait "${PIDS[$index]}"; then
          unset 'PIDS[index]'
          echo "[ERROR] ${stage} worker failed; rerun unchanged to resume. Log tails:" >&2
          for shard in "${!GPUS[@]}"; do tail -n 70 "${WORK_DIR}/${stage}_shard_${shard}.log" >&2; done
          exit 1
        fi
        unset 'PIDS[index]'
      fi
    done
    if ((${#PIDS[@]})); then sleep 1; fi
  done
}
run_workers export
CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" mara_ppo_experiment.py --stage audit --work_dir "${WORK_DIR}" --cpu_threads "${CPU_THREADS}"
run_workers train
CUDA_VISIBLE_DEVICES="" "${TRAIN_PYTHON}" mara_ppo_experiment.py --stage collect --work_dir "${WORK_DIR}" --cpu_threads "${CPU_THREADS}"
echo "Finished. Upload only: ${WORK_DIR}/ppo_report.zip"
