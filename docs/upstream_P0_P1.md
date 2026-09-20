# 上游定位头 P0＋P1 修复实验

这是一条独立入口。旧 `upstream_localization.py`、旧头、旧 VLM/特征缓存及历史结果不修改。
只实现 P0＋P1；不加入 P2 空间匹配、不重新调用 Qwen、不接 MARA。

## 服务器一体化启动

先激活原模型训练环境，然后拉取代码。这里不需要激活或提供 VLM 环境。
`SOURCE_WORK_DIR` 必须是上一轮完成导出的完整实验目录，含 `config.json`、`manifest.json`、`sealed.json` 和 `features/`，不能仅指向 metrics 文件夹。

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main

nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  SOURCE_WORK_DIR=./checkpoint/upstream_localization_v1 \
  WORK_DIR=./checkpoint/upstream_P0_P1_v1 \
  GPU_IDS=0,1 \
  HEAD_EPOCHS=15 \
  HEAD_BATCH_SIZE=4 \
  bash run_exp_upstream_repair.sh \
  > nohup_upstream_P0_P1_v1.out 2>&1 &

tail -f nohup_upstream_P0_P1_v1.out
```

如果上一轮目录不同，只改 `SOURCE_WORK_DIR` 为实际路径。源目录和新目录不能相同，也不能互为子目录。
流程：校验封存缓存 → 双卡分类别 P0 → 纯视觉头双卡 DDP 训练 → VLM 头双卡 DDP 训练 → 双卡分类别测试。
每卡 batch=4，全局 batch=8。源域选模由 rank 0 执行，期间另一张卡等待；不代表 DDP 训练只用了单卡。
这版不用加载 DINO、CLIP 或 Qwen 权重，但原数据 RGB/mask 路径必须仍有效，以验证输入一致性和训练/评价。
校验缓存会读取全部特征计算 SHA，可能需要一些时间。无需再生成多轮 VLM 描述，也无需复制几十 GB 特征。

只运行 P0：在 nohup 参数中加 `RUN_TRAIN=0 RUN_EVAL=0`。
已完成 P0 后续跑：同一命令加 `RUN_P0=0`。epoch 中断会从最后一个完整 epoch 的优化器状态继续。
配置/修复代码变化要换新 `WORK_DIR`，仍可复用同一旧 `SOURCE_WORK_DIR`。

## P0：定位数值问题，而非预先认定根因

`p0/metrics.csv` 同批比较：

- `base`：原始缓存概率。
- `legacy_clamp`：仅执行旧 `[1e-5,1-1e-5]` 截断，隔离其排序损失。
- `zero_residual`：新定位头零初始化后的实际前向输出，必须与 Base 逐元素完全一致，否则停止。
- `old_visual/old_vlm`：若旧目录存在匹配的 `heads/<mode>/best.pt`，复现旧训练头输出，记录 GT/背景上的平均 logit 修正、背景概率增量。这些仅用于诊断，不参与新头选模。

`p0/probability_audit.csv` 记录正常/缺陷像素中低于旧截断下限的数量，以及精确 0/1 概率数量。
PRO 与 AUC 用原有 `map_metrics` 在完整评价图上计算。P0 包括 source val 和 target eval，以 `partition` 区分。
不能把源域验证结果称为独立泛化成绩，旧 Base 可能已经见过源域验证图像。
P0 的目标标签用于事后诊断，不进入 P1 训练、alpha 选择或 checkpoint 选择。

## 数值修复

不对所有低概率像素抬底。用 float64、`expm1` 和非正指数计算 odds 修正的“变化量”，再加到原始概率上。
对任意缓存的内部概率，等价于 `sigmoid(logit(p)+alpha*delta)`，但零修正不经过会改变排序的截断/往返变换。
`alpha=0` 直接返回 Base；`delta=0` 也严格返回原始概率，同时保留初始梯度。

精确 0/1 没有有限 logit。仅对这两个端点使用 `1e-6/1-1e-6` 的有限代理计算变化量，保留原始概率作为起点，最终限制到 [0,1]。
因此不会给零残差输出加上代理偏置，端点也不是永久锁死、无法恢复的区域。
BCE 在稳定 logit 空间计算，精确端点采用该有限代理损失；Dice 和误报约束使用实际输出概率。这个端点处理与内部非端点的精确 odds 变换有区别。

## P1：训练目标

架构与语义输入保持 v1 一致，只改输出计算、训练约束及选模。纯视觉/VLM 两个头均从相同随机种子重新初始化，不从已经退化的旧头续训。

每图损失：

`BCE + positive-image Dice + BG_WEIGHT × hard-background-increase + RESIDUAL_WEIGHT × SmoothL1(delta,0)`

- hard-background-increase：源域 GT 正常像素中，预测相对 Base 的正向增量最大的 1% 的平均值。
- 正常区域错误抬分会受惩罚；GT 缺陷区域不受该背景保持项约束。
- 修正幅度软约束鼓励少改，不设置 Base 高分候选门控，也不设置残差幅度硬上限。
- 默认 `BG_WEIGHT=1`、`RESIDUAL_WEIGHT=0.01`、`HARD_BG_FRACTION=0.01`，是预先设定的探索性参数，尚无真实实验收益保证。

## 源域选模与回退

只读取旧 manifest 的 `train/val` 分区，沿用上一轮的监督协议；目标评价记录不能进入训练/选模。
候选强度 `REPAIR_ALPHAS=0.1,0.25,0.5,1`，并始终保留 alpha=0 的 Base。先确定配置，再看目标测试结果。

每个 epoch 在源域验证集中：

1. 每图固定随机抽取最多 `VAL_PIXELS=8192` 个像素，抽样不依赖标签。不同模式、epoch、alpha 使用相同抽样位置。
2. 每类别从抽样背景设置 FPR≤`VAL_FPR=0.01` 的保守阈值（严格 `>`，分数相同时实际 FPR 可能更低）。
3. 在完整分辨率 GT 连通域上计算平均区域召回；小缺陷定义面积≤图像 0.1%，命中要求该区域至少 10% 像素超过阈值。
4. 候选须满足：宏平均区域召回和小缺陷命中率不低于 Base；抽样像素 AUC 比 Base 下降不超过 `VAL_AUC_TOLERANCE=0.002`；每类别在 Base 阈值下的抽样背景 FPR 不超过 Base＋`VAL_FPR_SLACK=0.002`。
5. 合格候选按区域召回和小缺陷命中率的均值选优；没有小缺陷时只用区域召回。只有严格优于当前最佳才更新，平分保持更早选项；alpha 按从小到大检查。

上述数值是比例：0.002=0.2 个百分点。源域选模使用有界成本的抽样 AUC/FPR 和完整 GT 区域，不冒充全分辨率精确 PRO。
若抽样没有正例/背景，明确报错，不能默默视为零；用新目录增大 `VAL_PIXELS`。
`best.pt` 最开始就是零修正 Base，`last.pt` 保存实际训练进度。两者用途不同。
即使源域符合约束，也**不保证目标域不退化**，这不是部署安全保证。

## 输出与判读

```text
upstream_P0_P1_v1/
  repair_config.json
  p0/metrics.csv
  p0/probability_audit.csv
  heads/visual/{best.pt,last.pt,history.json}
  heads/vlm/{best.pt,last.pt,history.json}
  results/metrics.csv
  results/summary.json
```

`results/metrics.csv` 比较 Base 与源域选定的 visual/vlm：

- `alpha`：源域选定强度。
- `selected_epoch`：0 表示未采用训练修正。
- `selection=BASE_FALLBACK`：没有通过源域收益/误报要求的候选；输出相同是明确回退，**不是涨点，也不是没运行 VLM 分支**。
- `selection=SOURCE_VALIDATED`：通过源域检查的训练头，仍需观察目标成绩。
- PRO_exact/P_AUROC/I_AUROC/F1_best：沿用完整分辨率评价口径；F1_best 只是目标阈值扫描报告，不用于部署或选模。
- `_at_target_fpr_DIAGNOSTIC`：评价分区标签下、抽样背景 FPR≤1% 的区域/小缺陷指标，仅作 ROC 式评价，不是可部署阈值，不参与训练。P0 的 `partition=val` 行表示对应源域验证分区，同样仅供诊断。
- 固定 0.5 下的小缺陷恢复/退化、背景误报及修正统计也保留。

如果全部回退：先看 `history.json` 中每个 alpha 的验证指标和 `eligible`，不要为了得到非零改动而直接放开约束。
如果 P0 显示截断损失很小：不能将旧头退化归因于截断，继续检查旧头修正统计和 P1 的域外效果。
不要在这批目标标签上不断选超参数后，把同一批成绩作为独立泛化结果。

## 验证范围

自动测试包含极低/次正规概率、0/1 端点、极端残差、零残差精确相等与梯度、截断导致 AUC 损失的反例、背景约束、选模拒绝、Base 回退、训练标签隔离、断点恢复、旧缓存逐文件不变及两卡 shell/DDP 启动。
这些是合成数据和接口测试，不是两张 4090 上的真实精度实验。服务器实验结束后再判断收益。
