# 三阶段对齐实验：运行与结果

实现入口是 `run_exp_alignment_stages.sh`，协议见 `stage1_alignment_three_stage_plan.md`。
新代码不修改旧 `reference_highres_experiment.py`、`tools/reference_highres.py` 或 v1 封存目录。
仅使用本地现有 Base/DINO/CLIP 权重，不下载模型，不训练 RL/VLM。

## 必需输入

`V1_DIR` 指向上轮**完整实验目录**，例如 `checkpoint/reference_highres_v1`，而不是其 `results/`
子目录或下载的结果 ZIP。必须保留 `config.json`、`manifest.json`、`references.json`、`sealed.json`、
`anchors.npz`、`features/*.npz/json`，以及配置引用的原始图像、源域 mask、目标评价 mask、Base 与 DINO 权重。
旧代码/配置/权重/数据身份不一致会明确停止，不能通过手改 seal 绕过。
两个目录须独立；新实验不要覆盖旧实验。

沿用 `qfg_addino` 环境，需要原项目的 torch、numpy、Pillow、scipy、scikit-learn、cv2，
以及 Bash/util-linux 的 flock。脚本不安装或升级包。

## 2026-10-26 修复：从 residual parity 报错处恢复

旧版 `2f826ad` 在 T 阶段补导出时重新计算了未被 T 使用的参考残差，并用统一 `1e-3`
门限与旧缓存比较。离散 top-k 匹配的小变化可能放大残差差异；仅凭原报错不能确认服务器差异的具体来源。
修复后 T 只补导出 `query/prior/base` 和 source 教师，完全不做参考图编码或重新匹配。
R1/R2/R3 仍使用同一份封存 v1 残差、距离与熵，参考实验的证据没有改变。
这不是简单放宽 residual 门限：新代码不再生成无用的 residual；实际使用的视觉/Base 仍检查一致性。
Base 保持 `1e-5` 预算，视觉特征保留 `1e-3` 预算并按缓存数值增加 FP16 半 ULP 舍入容差。
每张图的六项检查均记录 max/mean/p99/RMSE 和失败元素数，即使失败也保留诊断。
另修复完整三阶段运行时前端选择仅纳入 H3 的条件错误：阶段三现在同样从 H3/T10/T01/T11
的源域合格候选中锁定最优前端，而非忽略已经完成的 T 实验。

如果你在 H 完成后、source extra 导出未封存时遇到该报错，先确认旧进程已经退出，再执行：

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main
conda activate qfg_addino

nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  V1_DIR=./checkpoint/reference_highres_v1 \
  WORK_DIR=./checkpoint/alignment_three_stage_v2 \
  GPU_IDS=0,1 TRAIN_SEEDS=42 \
  HEAD_EPOCHS=15 TRAIN_UNTIL_EPOCH=5 \
  REPAIR_EXPORT=1 START_PHASE=2 \
  bash run_exp_alignment_stages.sh \
  > nohup_alignment_three_stage_v2_repair.out 2>&1 &

tail -f nohup_alignment_three_stage_v2_repair.out
```

保持目录、workers、种子、计划轮数、损失权重等与原命令一致。
`REPAIR_EXPORT=1` 只接受已知旧版的精确代码哈希和未改变的 H 模型/损失代码：

- 旧 `config.json` 和全部 H checkpoint 保持原始字节，不改 fingerprint，不重训 H。
- 未封存 `extra/` 原子移动到 `recovery/extra_before_visualfix1/`，可恢复，不删除。
- `export_repair.json` 记录旧配置哈希、实际新代码哈希、保留权重哈希和迁移状态。
- 新的 source extra 单独重新导出封存；不会将两种格式混用，也不会复制 v1 大缓存。
- 已有 T/R 训练、extra seal、评价报告或未知代码版本会拒绝自动迁移，要求独立新实验目录。
- 同一修复重复执行不会再次移动新缓存；后续续跑可保留 `REPAIR_EXPORT=1`。
- 若进程在缓存移动时中断，可用相同命令重试，prepared/complete 记录负责恢复。

修复后的新实验直接按下一节启动，无需 `REPAIR_EXPORT=1`。

## 首轮：两张 GPU、一个训练种子、先 5 轮

将本次新文件同步到服务器同一项目目录后：

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
conda activate qfg_addino

nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  V1_DIR=./checkpoint/reference_highres_v1 \
  WORK_DIR=./checkpoint/alignment_three_stage_v2 \
  GPU_IDS=0,1 TRAIN_SEEDS=42 \
  HEAD_EPOCHS=15 TRAIN_UNTIL_EPOCH=5 \
  bash run_exp_alignment_stages.sh \
  > nohup_alignment_three_stage_v2.out 2>&1 &

tail -f nohup_alignment_three_stage_v2.out
```

每张卡一个独立训练或导出 worker，并不是 DDP。默认流程：

1. 检查旧配置、代码/权重身份与独立新目录。
2. 用旧缓存训练 H1/H2/H3；H2 真正删除新增文本分支，H3 移除隐藏特征直通修正。
3. 只用源域结果判断是否进入阶段二。通过后，首张真实图及后续每张图均对 global/high 的
   query/prior/base 做数值一致性检查，再分片补导出 source 的逐视图视觉证据与 768 维冻结教师，
   封存后训练 T10/T01/T11；T00 复用 H3。T 导出不编码参考图，不做 top-k 匹配。
4. 只用 source 锁定前端。拟合 source TRAIN 正常像素的距离中位数/IQR，并在 source val 校准 R1。
5. 前端合格时训练 R2/R3；不合格时保留冻结 Base 和 R1 诊断，明确跳过 R2/R3，不把跳过当程序失败。
6. 输出 source margin AP/AUROC、正常特征漂移和修正参数诊断。
7. 如需要晚融合，分片补导出目标逐视图特征（不导出目标教师，不读目标 mask），然后评价各组的
   primary 和 bce 两套预先固定选择器，汇总结果和可视化。
8. 默认每个主方法做少量真实在线端到端计时，参考池预编码成本单列；可关闭以缩短筛选。

目标 mask 只在所有模式预测后进入评价。源域 PRO/AP Pareto 主规则不合格会回退。
旧 BCE+Dice 选择器仅作诊断，不从两套中挑目标分数更高的一套作为主结果。

## 分阶段、续跑与多种子

- 只先做 H：`STOP_PHASE=1 RUN_BENCHMARK=0`。之后用相同目录恢复默认 STOP_PHASE=3 即可继续。
- `START_PHASE=2/3` 跳过已完成的前面训练，要求前面的必要对照已经完成当前指定轮数。
- 继续到 15 轮：保持 `HEAD_EPOCHS=15`、旧/新目录和其他配置不变，只改 `TRAIN_UNTIL_EPOCH=15`。
  关键对照同步续跑，不拿 15 轮候选对比 5 轮对照。
- 若续跑改变了所选前端，R2/R3 自动使用新的父权重哈希目录，从头训练到指定轮数；
  不会把旧前端上的参考分支当成可恢复断点。这种情况下阶段三会额外耗时。
- 正式重复：`TRAIN_SEEDS=42,43,44`。只改变新头初始化；数据划分、参考池和投影沿用 v1 的固定种子。
  这是完整关键对照入口，不会依据目标成绩自动剔除对照。
- 可选辅助损失消融：`RUN_AUX0=1` 增加 H1_aux0（保留文本通路，仅关闭辅助损失）。
  若曾启用，后续匹配轮次的续训也需保持启用，否则会因该对照轮数不一致而拒绝汇总。
- 只评价已有训练：`RUN_TRAIN=0`，指定与已完成训练完全一致的轮数；所需 extra 缓存仍会检查。
- 已有全部 extra 缓存时可设 `RUN_EXTRA_EXPORT=0`，但仍会核验封存内容，缺失则停止。
- 明确希望在源域回退后仍强制做机制实验时，设 `CONTINUE_ON_SOURCE_FALLBACK=1`；
  强制标志写入决策文件，但不会取消主选模的回退或改变训练标签政策。
- 在线计时：`RUN_BENCHMARK=0/1`、`BENCHMARK_IMAGES=4`。更多图像的稳定时延需单独扩大样本。

不要把已训练到 15 轮的 best/last 用 `TRAIN_UNTIL_EPOCH=5` 报成 5 轮结果，程序会拒绝。
每轮均保存 checkpoint 与验证表，可追溯 1~15 轮，但不使用目标最佳轮选模。
新的训练损失/代码/模型配置需新 WORK_DIR；只有上述已知 H-only 版本有显式审计修复入口。
旧封存文件不重写。

## 输出与阅读顺序

```text
alignment_three_stage_v2/
  config.json
  extra/                         # 逐视图 query/prior/base、source teacher；坐标在 JSON
  parity/source|eval/            # 每图 global/high 视觉/Base 的六项误差诊断
  export_repair.json             # 仅从已知旧版显式恢复时存在
  recovery/extra_before_visualfix1/ # 仅恢复时存在：原未封存 extra 的可恢复备份
  extra_source_seal.json
  extra_eval_seal.json
  decisions/seed42_*.json        # 源域阶段门、锁定前端、强制标志
  distance_stats.json            # 仅源域训练正常像素拟合
  runs/seed42/heads/H1/...       # epoch_*.pt, best_primary/bce.pt, last.pt, history
  runs/seed42/reference_<hash>/  # 与锁定前端绑定的 R1/R2/R3
  diagnostics/seed42/            # source margin、教师漂移、修正参数
  panels/seed42/through5/        # 哈希选样，含查询/正常参考/GT/各组概率与距离/熵
  results/seed42/through5/       # 逐类别、逐选择器评价
  reports/through5/
    README.md
    metrics.csv
    summary.csv
    comparisons.csv
    source_screen.csv
    seed_summary.csv
    end_to_end_seed42.csv
```

先看 `source_screen.csv` 的 fallback、epoch、alpha，再看 `comparisons.csv`：
H2-H1、H3-H1；T10-T00、T01-T00、T11-T10、T11-T01 及交互；R1-R0、R2-R0、R3-R2。
T00 只是 H3 的别名，不能计为独立重复。回退的 R2/R3 是回退到锁定前端，不一定是最初的冻结 Base。
三种子标准差不足两个种子时留空，不填零；不同数据集分别做类别宏平均。

`small_hits_target_fpr1_EVAL_ONLY` 使用目标固定误报预算，仅用于公平诊断，绝不是部署阈值。
`head_seconds_per_query` 只含缓存证据上的头与必要前端头，不含 DINO；端到端速度看独立的
`end_to_end_seed*.csv`，它包含解码、全局/四块 DINO、需要时的匹配和小头，排除一次 warmup。
参考池建立成本单列；少量图像的时间不是完整数据集吞吐结论。
history 中 FLOPs 只统计当前头 Conv2d 的前向运算，不包含 backbone、归一化或反向传播，
不能当完整训练 FLOPs。这里明确标注范围，不推测不可验证的完整运算量。
训练 BCE 对精确 0/1 使用有限端点代理；晚融合代理以 float64 拼接并扣除初始偏移，
保留端点梯度与零修正的 Base 损失。实际输出不加概率底值，零修正保持原图不变。

逐视图缓存会额外占用磁盘，source 教师尤其明显；不自动复制旧大缓存。
prepare 会显示全阶段额外缓存估算与当前剩余磁盘；阶段检查跳过后实际占用可能更少。
旧模型按 512/64/768 默认形状时，一个 source 查询五视图教师约 7.5 MiB（FP16，仅教师张量），
还需其他证据；实际以文件与磁盘为准。目标缓存不保存教师。
extra 不再保存或重算 residual/distance/uncertainty，也不重新生成 top-k 对应坐标。
可视化中的正常参考、距离与熵继续来自 v1 封存记录，不能声称是新匹配结果。

## 本地验证与实际限制

测试入口：`python -m unittest tests.test_alignment_stages -v`。
Windows 可设 `ALIGNMENT_TEST_BASH` 为 Git Bash 的绝对路径以测试双卡调度、失败停止和主动跳过。
合成数据流程测试覆盖八条训练路径、精确恢复、teacher detach、微小缺陷正常单元排除、
旧 H1 非零输出/损失一致性、缓存/参考身份、目标标签隔离、两套选模、报告过期检测及在线计时路径。
修复回归另覆盖近并列 top-k 残差跳变、完全禁止 T 导出重新匹配、Base 真实漂移拒绝、
全字段诊断持久化，以及旧 H checkpoint 字节不变的迁移与续训、迁移中断恢复和拒绝未知版本。
这些不是服务器真实工业精度实验，也不替代真实 DINO/CLIP 权重、BF16 与双 GPU 显存验证。
