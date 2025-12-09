
# python train.py --result_path ./checkpoint --device cuda:0 --dataset mvtec
# python train.py --result_path ./checkpoint/learnablePrompt --device cuda:0 --dataset mvtec
#!/usr/bin/env bash
set -e  # 只要有一条命令报错就退出脚本，避免训练失败还继续测试

# ====== 训练部分 ======
RESULT_PATH="./checkpoint/learnablePrompt"
DEVICE="cuda:0"
TRAIN_DATASET="mvtec"

echo "===== Start Training on ${TRAIN_DATASET} ====="
python train.py \
  --result_path "${RESULT_PATH}" \
  --device "${DEVICE}" \
  --dataset "${TRAIN_DATASET}"

echo "===== Training Finished ====="


# ====== 测试部分 ======
declare -a DATASETS=("mvtec" "visa")
SAVE_PATH="./TESTING_ALL/learnablePrompt"

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
