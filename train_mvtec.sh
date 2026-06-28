
# python train.py --result_path ./checkpoint --device cuda:0 --dataset mvtec
# python train.py --result_path ./checkpoint/learnablePrompt --device cuda:0 --dataset mvtec
#!/usr/bin/env bash
set -e  # 只要有一条命令报错就退出脚本，避免训练失败还继续测试

# 1) 日志目录与文件名
LOG_DIR="./logs"              # 改成你希望保存的目录
# TRICK_NAME="base"  # 改成你希望的trick名称
TRICK_NAME="learnablePrompt"  # 改成你希望的trick名称
EPOCH=15
BS=16
DIS="测试长度20，冻结全局seg"
mkdir -p "$LOG_DIR"

# 用脚本名 + 时间戳，避免覆盖
SCRIPT_NAME="$(basename "$0" .sh)"
TS="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/${TRICK_NAME}/${SCRIPT_NAME}_${TS}_${DIS}.log"

# 2) 把标准输出/错误都重定向到日志（追加写入用 >>，覆盖用 >）
exec >>"$LOG_FILE" 2>&1

echo "[$(date +'%F %T')] Start. pid=$$"
echo "Log: $LOG_FILE"


# ====== 训练部分 ======
RESULT_PATH="./checkpoint/${TRICK_NAME}/${TS}"
DEVICE="cuda:1"
TRAIN_DATASET="mvtec"

echo "===== Start Training on ${TRAIN_DATASET} ====="
echo "Results will be saved to ${RESULT_PATH}"
python train.py \
  --result_path "${RESULT_PATH}" \
  --device "${DEVICE}" \
  --epoch "${EPOCH}" \
  --batch_size "${BS}" \
  --dataset "${TRAIN_DATASET}"

echo "===== Training Finished ====="

# ====== 画 loss ======
LOSS_FILE="${RESULT_PATH}/loss.txt"
if [ -f "${LOSS_FILE}" ]; then
  echo "===== Plot Loss: ${LOSS_FILE} ====="
  python plot_loss.py --path "${LOSS_FILE}"
else
  echo "[WARN] loss file not found: ${LOSS_FILE}"
fi

# ====== 测试部分 ======
declare -a DATASETS=("visa" "btad" "mpdd")
SAVE_PATH="./TESTING_ALL/${TRICK_NAME}/${TS}"

for ds in "${DATASETS[@]}"; do
  echo "===== Start Testing on ${ds} ====="
  python test2.py \
    --result_path "${SAVE_PATH}" \
    --weight_path "${RESULT_PATH}/ckpt" \
    --device "${DEVICE}" \
    --dataset "${ds}"
  echo "===== Testing on ${ds} Done ====="
done

# ====== 画各数据集 metric ======
for ds in "${DATASETS[@]}"; do
  METRIC_FILE="${SAVE_PATH}/${ds}/metric.txt"
  if [ -f "${METRIC_FILE}" ]; then
    echo "===== Plot Metric: ${METRIC_FILE} ====="
    python plot_metric.py --path "${METRIC_FILE}"
  else
    echo "[WARN] metric file not found: ${METRIC_FILE}"
  fi
done

# ===== 汇总数据 ====
SUMMARIZE_METRICS="${TRICK_NAME}/${TS}"
echo "===== Plot SUMMARIZE_Metric: ${SUMMARIZE_METRICS} ====="
python summarize_metrics.py --path "${SUMMARIZE_METRICS}"
echo "===== ALL DONE ====="
