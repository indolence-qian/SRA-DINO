# P2 加速实验：先完成六组筛选，再续跑

本入口保持 P2 的六组设计：P2a/P2b × visual/real/shuffled。P2a 保留 P1 损失，P2b 增加区域平衡和缺陷—邻近背景排序。它解决执行效率问题，性能是否改善仍需真实实验结果。

## 显存低的原因与执行调整

原入口只训练小定位头，DINO/CLIP/Qwen 已经缓存，因此模型参数显存本来就小。六组依次执行双卡 DDP，rank 0 做源验证时另一张卡等待；NPZ 解压、掩码处理、连通域计算、GPU 标量读取也可能使 GPU 等待。低显存占用不能单独证明算力瓶颈在哪里，新的日志会记录吞吐与各阶段耗时。

加速入口使用 **两张卡各一个长期存活的独立 worker**，通过任务队列动态领取六个对照，先处理 P2a 真实/打乱/纯视觉，再处理 P2b。每卡同时训练一个头，避免 DDP 验证屏障；完成较快的 worker 继续领取下一项。所有组采用相同初始化种子、相同每轮样本顺序、相同有效 batch、相同精度协议和完整源验证。

缓存和加速措施：

- 只读原封存缓存；第一次将 **source train/val** 转为派生的未压缩张量包，避免每轮反复解压 NPZ。目标不进入训练包。
- mmap 提供 CPU/磁盘后备，各 worker 根据启动时空闲显存，扣除默认 4 GiB 后，再使用剩余量的 85% 为数据缓存预算。`GPU_CACHE_GB=0` 表示自动，也可设置上限。缓存不足时使用部分常驻和后备读取。
- 真实和打乱文本的匹配图每个 worker 只计算一次，保持 FP32；同类、同 partition、其他图像 donor 的规则不变。
- 源 GT 连通区域和每个区域的背景环预计算；训练时避免按组件反复读取 CUDA 标量。
- 源验证的随机采样位置和全部 GT 组件索引预计算。每轮只传输抽样分数及 GT 像素分数，用分组计数计算区域召回，保留原采样、阈值、8 连通区域和选择门控。
- 4090 默认使用 BF16 autocast 运行定位头。概率修正、loss 的 logit 运算保持稳定实现。六个对照全部使用同一 BF16 协议。

原 DDP 默认全局 batch 是 2×4=8；新入口默认单卡有效 batch=8。micro-batch 默认 8；训练 OOM 时自动减半并累积梯度，有效 batch 和 optimizer 更新频率保持不变。模型结构包含 GroupNorm，没有跨样本的 BatchNorm。

目标推理也会在 OOM 后缩小 micro-batch，保留全量样本。若 optimizer 更新时发生 OOM，停止当前运行，避免重试可能已部分更新的状态；缩小缓存预算后可从 last 续跑。

新的代码和数据不写入原 P2 目录，也不修改原 P2 的哈希绑定文件。因此旧实验可以保留、继续跑；新结果与旧 FP32/DDP 结果分开记录。BF16、样本次序和浮点归约方式有数值差异，不能宣称与原运行逐位相同。

## 赶进度的实验安排

第一轮将六组都跑到 **5 个 epoch**，完整源验证每轮执行，不删减训练样本。使用 source val 选择 best 和 alpha，包括 Base 回退；目标依然全量评估。

为缩短目标评估，只评估 **Base + 六组 source-selected best**。默认省略 last、alpha=1 和同头推理干预的大量重复评估；已有语义审计可供参考。每个类别的原 NPZ 只读一次，真实/打乱匹配在各头之间复用。

5 轮结果用于方向筛选，不能称为 15 轮充分训练的结果。完整计划仍是 15 轮；将 `TRAIN_UNTIL_EPOCH` 改成 15，保持其他配置和 `WORK_DIR` 不变，即从 5 轮继续训练，保留 optimizer 和已选 best。重复同一停止轮次不会重新训练。提升停止轮次后，评估检测到检查点变化会重算。

当前 MVTec 已用于多次诊断；实验设计和 epoch/alpha 选择依然只依据源数据。多种子和新的独立目标验证在方向明确后开展。

## 运行

默认优先复用 `DIAGNOSTIC_WORK_DIR` 中已完成的 P2 质量/匹配材料，并核对封存来源和相关参数，不重写原报告。如果没有可用报告，自动在新目录完成这两个阶段。人工语义准确率没有填写时仍保持 `NOT_MEASURED`。

```bash
conda activate qfg_addino
cd /mnt/qfg/Tate_qfg/SRA-DINO

nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  SOURCE_WORK_DIR=./checkpoint/upstream_localization_v1 \
  DIAGNOSTIC_WORK_DIR=./checkpoint/upstream_P2_v1 \
  WORK_DIR=./checkpoint/upstream_P2_fast_v1 \
  GPU_IDS=0,1 \
  HEAD_EPOCHS=15 \
  TRAIN_UNTIL_EPOCH=5 \
  EFFECTIVE_BATCH_SIZE=8 \
  MICRO_BATCH_SIZE=8 \
  HEAD_PRECISION=bf16 \
  GPU_RESERVE_GB=4 \
  CPU_THREADS=4 \
  PACK_WORKERS=4 \
  bash run_exp_upstream_p2_fast.sh \
  > nohup_upstream_P2_fast_v1.out 2>&1 &
```

源目录必须是含 features/reviews/config/sealed 的原始定位缓存，不是 P0/P1 或 audit 输出。无需启动 VLM 环境，不重跑 Qwen，不重新提取 DINO/CLIP 特征。

第一次的校验、派生张量包和 worker 初始化会有准备耗时；日志会标明阶段。张量包需要额外磁盘空间，`packed/complete.json` 记录实际字节数。准备阶段低显存属于正常情况。

如需完整 15 轮，复制同一命令，将 `TRAIN_UNTIL_EPOCH=5` 改成 `TRAIN_UNTIL_EPOCH=15`，日志文件可换成 `nohup_upstream_P2_fast_full.out`。不要改 `HEAD_EPOCHS`、精度、有效 batch、loss 参数或代码；这些变化必须换目录，不能混用检查点。

如只急着看源验证筛选，可加 `RUN_EVAL=0`；之后用相同配置加 `RUN_TRAIN=0 RUN_EVAL=1` 补全目标评估。只报告源验证不能代替目标泛化结果。

两张卡已经有其他任务时，自动缓存预算会参考当前空闲显存；不会停止那些任务。峰值变化仍可能触发 OOM，micro-batch 回退到 1 后仍不足则明确失败。可通过 `GPU_CACHE_GB` 限制缓存量；不要仅为显存百分比扩大模型或有效 batch。

## 结果和验证

默认服务器输出目录：`/mnt/qfg/Tate_qfg/SRA-DINO/checkpoint/upstream_P2_fast_v1/`。

- `source_screen.csv`：每完成一组更新源验证筛选结果，包含完成轮次、alpha、samples/s、训练/验证秒数、micro/effective batch、常驻缓存和 peak allocated GiB。
- `heads/{p2a,p2b}/{visual,real,shuffled}/history.json`：每轮 loss、选参和吞吐/显存记录；best/last 保存于同目录。
- `results/README.md`：六组实际完成轮次、精度协议、完整目标 best 结果。
- `results/comparisons.csv`：real 对 visual/shuffled、P2b 对 P2a、各头对 Base 的配对差值，单位为百分点。
- `packed/`：派生 source 张量和监督索引；原缓存不变。

本地合成 CPU 测试检查缓存匹配头的输出/梯度、向量化 P2 损失的值/梯度、源验证指标与原实现等价，检查源划分隔离、只读缓存、有效 batch 累积、断点续跑、独立 worker 队列和失败停止。双 4090 的吞吐和显存使用需以服务器日志为准，未宣称固定倍数加速。

CUDA worker 使用 spawn；精度采用 PyTorch autocast。参考：[PyTorch 2.9 AMP](https://docs.pytorch.org/docs/2.9/amp.html)、[CUDA 多进程说明](https://docs.pytorch.org/docs/2.9/notes/multiprocessing.html)。
