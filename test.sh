
# python train.py --result_path ./checkpoint --device cuda:0 --dataset mvtec
# python train.py --result_path ./checkpoint/learnablePrompt --device cuda:0 --dataset mvtec
#!/usr/bin/env bash
set -e  # 只要有一条命令报错就退出脚本，避免训练失败还继续测试

# 1) 日志目录与文件名
LOG_DIR="./logs"              # 改成你希望保存的目录
TRICK_NAME="learnablePrompt"  # 改成你希望的trick名称
mkdir -p "$LOG_DIR"

# 用脚本名 + 时间戳，避免覆盖
SCRIPT_NAME="$(basename "$0" .sh)"
TS="20251230_152312"
LOG_FILE="${LOG_DIR}/${TRICK_NAME}/${SCRIPT_NAME}_${TS}.log"

# 2) 把标准输出/错误都重定向到日志（追加写入用 >>，覆盖用 >）
exec >>"$LOG_FILE" 2>&1

echo "[$(date +'%F %T')] Start. pid=$$"
echo "Log: $LOG_FILE"


# ====== 训练部分 ======
RESULT_PATH="./checkpoint/${TRICK_NAME}/${TS}"
DEVICE="cuda:0"
TRAIN_DATASET="visa"

# echo "===== Start Training on ${TRAIN_DATASET} ====="
# echo "Results will be saved to ${RESULT_PATH}"
# python train.py \
#   --result_path "${RESULT_PATH}" \
#   --device "${DEVICE}" \
#   --dataset "${TRAIN_DATASET}"

# echo "===== Training Finished ====="


# ====== 测试部分 ======
declare -a DATASETS=("mvtec" "btad" "mpdd")
SAVE_PATH="./TESTING_ALL/${TRICK_NAME}/${TS}"

for ds in "${DATASETS[@]}"; do
  echo "===== Start Testing on ${ds} ====="
  python test.py \
    --result_path "${SAVE_PATH}" \
    --weight_path "${RESULT_PATH}/ckpt" \
    --device "${DEVICE}" \
    --dataset "${ds}"
  echo "===== Testing on ${ds} Done ====="
done

echo "===== ALL DONE ====="
