# MARA 实验 G：统一基础后的收益下界阈值扫描

本实验冻结同一个 MARA 检查点，不训练 Base/MARA，不调用 VLM，不改奖励系数。
双 4090 各自运行独立 worker，按完整类别分工；24 GB 显存分别使用，不作为一个 48 GB 池。

## 一条 nohup 命令

在原 qfg_addino 环境、项目目录中执行。以下使用已经确认存在的权重文件。
SOURCE_WORK_DIR 指向原 A/B 运行目录，里面必须包含 config.json、manifest.json 和 evidence/，
不是仅下载的 results/ 压缩包。它只读复用原 Base 缓存，不复用旧策略轨迹。

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main
nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  MARA_CKPT=./checkpoint/mara_visa_hfa3_20260908_225850/ckpt/mara_epoch_29.pth \
  SOURCE_WORK_DIR=./checkpoint/mara_early_stop_ab_v2 \
  WORK_DIR=./checkpoint/mara_gate_sweep_g_v1 \
  GPU_IDS=0,1 \
  EXPORT_BATCH_SIZE=8 SWEEP_BATCH_SIZE=8 \
  DECODE_WORKERS=2 METRIC_WORKERS=2 CPU_THREADS=4 \
  SAMPLES_PER_CATEGORY=8 EMPTY_GT_POLICY=exclude \
  bash run_exp_mara_gate_sweep.sh \
  >> nohup_mara_gate_sweep_g.out 2>&1 &
```

没有原缓存目录时，删掉 SOURCE_WORK_DIR 那一行。此时每卡批量加载 Base，
每张图只前向一次，所有 G 组复用同份输出。已有缓存校验匹配时完全跳过大模型加载。
缓存有损坏、checkpoint 或 Base 配置不同会报错，不会静默混用。

```bash
tail -f nohup_mara_gate_sweep_g.out
tail -f ./checkpoint/mara_gate_sweep_g_v1/sweep_shard_*.log
```

## 实验组

| 组别 | gain_safety_margin | 转移行为 | 收益接受门 |
| --- | --- | --- | --- |
| G0 | 0 | 修复后 | 保留 |
| G1 | -0.01 | 修复后 | 保留 |
| G2 | -0.03 | 修复后 | 保留 |
| G3 | -0.05 | 修复后 | 保留 |
| G4 | -0.10 | 修复后 | 保留 |
| G5_bypass | 不适用 | 修复后 | 完全绕过 |
| L0_legacy | 0 | 仅恢复旧版 no-op 转移 | 保留 |

所有组都保留策略自主 STOP、ROI/layer 选择器和空间融合门。
接受概率、成本、奖励定义、horizon 全部继承同一 checkpoint。默认 STEPS=0，
即继承保存的三步 horizon；显式覆盖时必须使用新的 WORK_DIR，并注明覆盖。
L0_legacy 用来隔离基础修复的影响，不能把它与 G0 的区别解释成阈值收益。

只改变预测下界执行阈值 gain_safety_margin。
不改 gain_accept_threshold：后者改变 gain 分类和 STOP 辅助监督标签。
当前训练 rollout 默认使用软接受；实验 G 运行生产推理的硬接受门，不做重训。

## 统一状态行为和数据审计

MARAConfig.freeze_noop_state 默认 True。操作 STOP、硬拒绝或 inactive 时，
next_logits、next_prob、累计融合门和图像级 logits 都保持原状态。
活跃状态的未执行候选仍保留，用于训练收益预测和事后评分。
字段没有新增模型参数，旧 checkpoint 无需转换；只有 L0_legacy 显式设置 False。
旧权重是旧训练得到的，本实验不宣称修复已改变历史训练过程。

导出后先审计：实际图像标签、512/128 网格 GT 像素数、原始 mask 非零像素及最大值。
异常标签但处理后 GT 为空时，默认 EMPTY_GT_POLICY=exclude，对所有组一致排除。
原 mask 已空和预处理/阈值后消失分别记录；绝不自动修标注或根据实验效果删图。
EMPTY_GT_POLICY=flag 可保留全部样本另作敏感性实验，但必须使用新的 WORK_DIR。
正常标签却有前景 GT 会直接报错。小缺陷仅在 128 网格消失时仍保留并记录。
因此若原 288 图中确实存在一个空 GT，本轮应分析 287 图；原 288 图指标不能直接相减，
请使用本轮同样本集的 Base、G0 和 L0_legacy。

## 加速与显存

1. prepare：固定完整实验配置、样本清单、checkpoint/DINO 文件哈希。
2. 两 GPU export：优先校验复用旧证据，否则 FP32 批量 Base 推理、CPU 并行解码。
3. CPU audit：统一样本集并记录排除原因，不向策略传 GT。
4. 两 GPU sweep：每卡仅加载小 MARA；默认 batch 8，依次计算全部阈值组。
5. 指标计算按类别提交到每个 worker 的 CPU 线程池，与后续类别的 GPU 推理重叠。
6. collect：校验所有类别完整，汇总并生成小报告压缩包。

没有 AMP/量化/TF32 自动切换。OOM 时仅把当前批量减半并重试同一批，
不会跳样本、改变有效样本集或降低输入分辨率。最低 batch=1 仍 OOM 则明确退出。
默认每类 8 张，单类别批量上限也就是实际可用张数；没有必要为占满显存重复计算。
GPU 峰值和实际批量写入 resources_shard_*.json，CPU 指标阶段 GPU 暂时空闲可能正常。
旧证据保留原 FP32 值；新生成 Base 使用本轮记录的 FP32 配置。批量核函数与旧单图
运行可能有浮点差异，不把本轮 legacy 组声称为旧运行的逐位复刻。

原证据通过硬链接复用，跨磁盘时复制；不会改写原目录。每组轨迹只推理一次，
不重新跑 Base；相同预测的指标在类别内复用。预测图仅暂存 CPU 内存，
不保存大型逐图 NPZ，因此报告打包无需携带数 GB 中间文件。

## 结果和判断

完成后只需上传 WORK_DIR/gate_report.zip。包含配置、清单、审计、完成标记和：

- results/macro_metrics.csv：各数据集每步类别宏平均，优先看 PRO/P-AUROC/误报。
- results/metrics.csv：每类别、每步指标，空值不是 0。
- results/gate_summary.csv：正常/异常分开，执行数、正收益误拒、负收益执行。
- results/step_trace.csv：逐样本逐步动作、门控、质量分量、成本、奖励和累计回报。
- results/comparisons.csv：各组相对 G0 的配对类别变化。
- results/foundation_comparison.csv：L0_legacy 相对修复后 G0 的变化。
- results/data_audit.csv：原始/预处理 GT、排除原因和最终统一样本集。
- resources_shard_*.json：各卡实际 batch 上限、峰值显存与运行时间。

所有指标直接来自同一条三步轨迹各步输出，不分别改 horizon 来比较前缀。
收益和执行次数上升不等于成功，必须同时检查定位和正常图误报。
这里只扫描预先声明的完整曲线，不按目标域挑最佳阈值或 checkpoint。
VisA 图像可能被旧 Base/MARA 训练见过，不能声称独立源域验证；每类 8 图仅作机制诊断。

同一提交、同一参数和 WORK_DIR 可以续跑；checkpoint、源文件、缓存、配置和
类别结果都校验。改变代码版本、阈值网格、空 GT 策略或采样必须换新目录。
单目录 flock 防止重复并发；任何 worker 失败会停止后续阶段并在主日志打印子日志末尾。
没有自动选择新权重、自动安装依赖、修改奖励或调用 VLM。
