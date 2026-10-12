# MARA 早停诊断：实验 A 与实验 B

入口：run_exp_mara_diagnose.sh。只读取一个明确的 MARA 权重文件，冻结所有权重，
不重训 Base/MARA，不调用 VLM，不选择目标域最佳检查点或阈值。
生产 STOP 转移行为保持原样；诊断会记录它是否改变图像级 logits。

## 一体化 nohup

在原 SRA 模型环境中运行。MARA_CKPT 必须换成这轮真实的 MARA 文件，不能给目录，
也不能给 Base 权重。MARA checkpoint 已保存 Base 适配器和提示权重，
因此不必单独提供 BASE_CKPT；仍需要原 DINO 预训练文件与本地 dinov3。

    cd /mnt/qfg/Tate_qfg/SRA-DINO
    git pull --ff-only origin main
    nohup env \
      TRAIN_PYTHON="$(command -v python)" \
      MARA_CKPT=./checkpoint/你的MARA目录/ckpt/mara_epoch_029.pth \
      WORK_DIR=./checkpoint/mara_early_stop_ab_v1 \
      GPU_IDS=0,1 \
      DATASETS="visa mvtec btad mpdd" \
      SAMPLES_PER_CATEGORY=8 \
      GRADIENT_IMAGES=16 \
      STEPS=0 \
      bash run_exp_mara_diagnose.sh \
      > nohup_mara_early_stop_ab.out 2>&1 &

    tail -f nohup_mara_early_stop_ab.out

默认每类最多 8 张，按 mask 路径存在/不存在分层，缺额由另一层补充。
每张图对 D0-D3 都使用同一份 Base 缓存。SAMPLES_PER_CATEGORY=0 表示全量。
这是诊断子集，不是新的正式泛化成绩。VisA 原始 test 分区可能被旧 Base/MARA
训练见过，必须注明，不能把这部分称为严格未见源域验证。
目标域标签只用于诊断与指标计算，不参与动作选择或超参数选择。

数据默认继承 Datasets 注册表，可按实际位置覆盖，例如：

    DATA_ROOTS='{"visa":"/实际/VisA","mvtec":"/实际/mvtec"}'

DINO 文件搬家时设置 DINO_REPO_DIR 和 DINO_WEIGHTS；未设置时沿用 checkpoint 路径。
STEPS=0 保留权重中的 horizon，STEPS=3 明确把四组诊断都设为三步。
最终 config.json 保存原始与实际生效配置，避免现有 test_mara.py 从 checkpoint
恢复配置时忽略命令行步数的歧义。步数不同时不能直接混作相同实验。

## 实验 A：逐步追踪

results/step_trace.csv 保存所有四组、所有样本、所有步骤，包括：

- STOP/REFINE、ROI、证据层概率和归一化 ROI 坐标；
- active 前后状态、操作选择、执行权重和明确的停止原因；
- 平均收益、保守收益下界、接受概率和接受条件失败原因；
- 低分辨率和原分辨率的当前、候选、执行后质量；
- 分类、定位、背景分数及置信度质量分量；
- 成本、即时奖励、无成本奖励、折扣累计回报；
- GT 在原网格及决策网格的像素数，检查小缺陷是否消失；
- STOP 和 inactive 之后热图/图像级 logits 是否变化。

该 CSV 中的候选是当前策略选中 ROI/layer 的反事实修正，
不是遍历 16x4 动作得到的 GT Oracle。本次范围只有 A/B，不包含候选枚举和成本重训。
GT 评分发生在 trace_infer 返回之后，不反馈给下一步动作。
即使停止，仍走完原 infer 的剩余循环，真实保留并揭示原实现的 inactive 漂移。

results/gradient_probe.csv 与 gradient_trajectories.csv 对固定抽取的样本做：
在一个独立 MARA 克隆上复现预热后的软门控、采样 group、Base 锚和
GRPO replay，记录正/负/零优势与三个动作头的纯策略梯度。没有 optimizer.step，
没有修改原权重或正式轨迹。它回答“当前权重是否还能产生策略梯度”，
不能证明历史训练的 W_GRPO 已开启、实际发生过参数更新或预热已结束。
checkpoint 原训练损失权重/完整训练配置未全部保存，历史更新仍需原训练日志。
STOP 动作会屏蔽 ROI/layer 的策略梯度，因此全 STOP 时这两个梯度为零本身
不等于实现故障；应同时查看 op_head、采样操作和有效决策数。
GRADIENT_IMAGES=0 可关闭这一项。

## 实验 B：冻结权重 2x2 对照

| 组别 | 是否允许策略 STOP | 收益接受门 |
| --- | --- | --- |
| D0 | 原策略 | 原设置 |
| D1 | 原策略 | 完全绕过 |
| D2 | 每个仍 active 的步骤强制 REFINE | 原设置 |
| D3 | 每个仍 active 的步骤强制 REFINE | 完全绕过 |

四组都保留空间融合门控、原 ROI/layer 选择、原 Refiner 和同一个 horizon。
D2 仍可因收益拒绝而停止；D3 则继续到 horizon。这里不会把
disable_hard_gain_gate 的软接受错误当成完全绕过，也不是仅强制第一步。
若 checkpoint 本身禁用了收益门，D0/D1 和 D2/D3 可以相同，
这不是实验故障；实际配置已记录。

结果先看：

1. results/README.md 与 macro_metrics.csv：各数据集类别宏平均、每步及最终变化。
2. stop_summary.csv：策略 STOP、收益拒绝、候选成本翻负、停止后输出漂移。
3. comparisons.csv：D1/D2/D3 相对 D0 的配对类别差值。
4. step_trace.csv：定位具体样本和步骤，区分即时收益不足与门控误拒绝。

只有执行次数增加不算成功；需要检查 PRO、P-AUROC、背景误报和分步收益。
PRO 使用旧 test2 的 sampled-threshold 定义（默认 1000 个阈值、FPR 0.3），
通过预标记连通区域与排序加速，不混用其他实验的 exact PRO。
像素 AUROC/F1 对同一概率图计算，I-AUROC 使用 dataset 图像标签与 max-pixel 分数。
二分类指标在标签单一时为空，而不是把“不可计算”记成 0；宏平均列出有效类别数。
背景 FPR@0.5 是固定阈值诊断，不是目标域调参。

## 安全、资源与续跑

prepare -> 两卡 export -> 两卡 diagnose -> CPU collect。
每卡独立加载模型，不使用 DDP。Base 每张图只前向一次，四组复用缓存。
export 完成后大模型进程退出，再运行轻量 MARA 诊断。
CPU 汇总时 GPU 空闲是正常现象，worker 详细日志在 WORK_DIR 内。
原分辨率四组各步预测以压缩 npz 保存，全量运行需预留磁盘和 CPU 内存。
类别指标逐类计算，并复用相同预测的指标结果。

复用同一命令、同一 WORK_DIR 即可续跑；校验 checkpoint、源文件、缓存和
结果哈希，配置变化则拒绝覆盖。改变采样、horizon、checkpoint、GPU 数量时
应使用新的 WORK_DIR。单目录 flock 防止并发运行，任何 worker 失败会终止，
collect 拒绝不完整结果。没有安装依赖、下载 VLM、删除旧结果或自动 git pull。

可单独跳过阶段，但仍必须保持 prepare 配置相同：

    RUN_EXPORT=0 RUN_DIAGNOSE=1 RUN_COLLECT=1 ...

正式结论应写“冻结权重下的机制诊断”，不能从本次 A/B 推断降低训练成本
必然有效。若需要成本因果实验，应另分离 RL cost 和收益/STOP 标签 cost，
保持同初始权重与训练预算，再做受控重训。
