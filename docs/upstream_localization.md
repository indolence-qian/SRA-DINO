# 上游局部语义定位头实验（2026-09-17）

## 本轮要回答的问题

之前的固定 Base → 候选区域 → VLM 复核 → 固定修正实验继续保留，作为历史参考，不覆盖目录、不改旧入口。
本轮不是继续调整候选区域的增减分，也不加入 MARA。我们要区分：

1. 新的监督定位头本身能否改善冻结 Base？
2. 在相同训练数据、相同结构和初始化、相同训练步数下，VLM 的局部描述是否提供额外信息？
3. 改善是否发生在小缺陷和 Base 的漏检区域，而不只是扩大高分背景？

## 数据流与可训练部分

```text
原始 RGB 图像
 ├─ 冻结旧双塔：DINO + HFA + 已训练的 CLIP 空间适配器
 │    ├─ 四层空间特征 [4, D, H/16, W/16]
 │    └─ 冻结 CLIP 文本提示分支 → 原始 Base 异常图
 │
 └─ 与 Base 分数无关的全图覆盖裁剪（默认 3×3，边长占比 0.4）
      └─ 冻结 Qwen3-VL-8B-FP8：整图上下文 + 原图局部裁剪
           └─ 局部实际观察 + 一般正常结构预期 + 可见性
                └─ 冻结 CLIP 文本编码器 → 两类局部文本向量
                     └─ 按裁剪坐标对齐到 DINO 网格；重叠处有效向量取均值

空间特征 + 局部文本条件 + 语义有效性 + Base 概率图
 └─ 新训练定位头：1×1 投影 → 空间卷积融合 → 上采样解码
      └─ 全图像素 logit 残差 + Base logit → 新异常定位图
```

- **训练**：只有新定位头。冻结 DINO/HFA、原有适配器、CLIP、原提示词和 Qwen。
- **不替换 CLIP**：这一版验证“用 VLM 增强上游语义”，CLIP 仍负责编码文本。
- **不是硬伪标签**：VLM 的 `possible_defect/apparently_normal/uncertain` 仅记录用于审计，不直接作为像素标签；训练目标来自源域真实 mask。
- **不是 VLM 直接分割**：局部文字是空间条件；由视觉定位头在网格内学习像素定位，文字没有像素边界精度。
- **不受候选门控限制**：任何位置都可学习修正，包括 Base 极低分位置；没有“必须先进入候选才能修改”的限制，也没有固定残差幅度上限。
- 最后输出层零初始化，初始输出等于概率截断到 `[1e-5,1-1e-5]` 后的 Base；之后允许正负修正。稳定 BCE-with-logits 和梯度裁剪用于控制训练数值。
- 捕获的是**旧适配器输出的多层 DINO 特征**，不是 Qwen 隐层特征，也不是 CLIP 图像塔特征。

默认训练两个同结构对照：

| 结果名 | 含义 |
|---|---|
| `base` | 同一批目标图像上的冻结旧双塔 |
| `visual` | 新定位头，但语义输入/有效性全部置零 |
| `vlm` | 新定位头 + 全覆盖局部 VLM 文本条件 |
| `vlm_zero_semantics_DIAGNOSTIC` | 已训练 VLM 定位头在推理时移除文本，用于判断是否依赖语义；不是另一个独立训练对照 |

两个训练头都在两卡上依次训练，不是一卡一个模型；特征与语义只缓存一次。MARA 后续应以新定位头确实有收益为前提再接入。

## 数据协议：这不是目标域无监督训练

现有 Base 训练代码使用 **VisA 的 TEST 分区及其异常 mask**，因此本轮默认沿用这个源域监督设置。
将源域按“类别 × 是否有 mask”做确定性 80%/20% 的 head train/val 划分，**仅用源域验证损失选择新定位头 checkpoint**。

重要限制：旧 Base 可能已经训练过该源域验证子集。因此它只是新头的源域选模集，不能报告为独立泛化成绩，更不能把本设置称为 VisA 无监督异常检测。

默认目标域为 MVTec TEST，可换成未参与训练的其他数据集。源域与目标域名称必须不同；图像内容 SHA 检查阻止 train/val/target 跨分区完全重复图像。若曾用 MVTec 调参，需要另保留未用于选方案的数据集用于最终论文结论。

- VLM 请求只包含 RGB 图、类别名称与固定提示词；不包含路径、mask、Base 热图或源域标签。
- 准备阶段读取路径/文件指纹，源域划分使用 mask 是否存在；VLM/特征导出不读取 mask 内容。
- head 训练和选模只加载 source train/val 的 mask。目标 mask 仅在最后评价时加载。
- 图像 resize 保持旧双塔输入约定。mask 按历史约定：PIL 双线性缩放后，VisA `>0`，MVTec/BTAD/MPDD `>127`。
- 历史截屏可能涉及不同数据集/子集与 PRO 实现，**本轮以同批 Base/visual/vlm 的差值为准**，不能直接混用历史绝对值。

## 一体化启动（服务器）

先确认新代码已经同步到服务器，激活原训练环境。已有模型目录和 Base 不改动。

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO

nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  VLM_PYTHON=/root/miniconda3/envs/sra_vlm/bin/python \
  VLM_MODEL_ID=/mnt/qfg/Tate_qfg/models/Qwen3-VL-8B-Instruct-FP8 \
  BASE_CKPT=/mnt/qfg/Tate_qfg/SRA-DINO/checkpoint/base_visa_hfa3_20260908_225850/ckpt \
  WORK_DIR=./checkpoint/upstream_localization_v1 \
  GPU_IDS=0,1 \
  SOURCE_DATASET=visa \
  EVAL_DATASETS=mvtec \
  HEAD_MODES=visual,vlm \
  HEAD_EPOCHS=15 \
  HEAD_BATCH_SIZE=4 \
  bash run_exp_upstream_localization.sh \
  > nohup_upstream_localization_v1.out 2>&1 &

tail -f nohup_upstream_localization_v1.out
```

`HEAD_BATCH_SIZE=4` 是每卡 batch，两卡全局 batch=8。脚本用 `torch.distributed.run --nproc_per_node=2` 启动实际 DDP 定位头训练。
开始前先对一张 RGB 图执行真实冻结双塔前向与文本编码预检，检查旧权重、依赖和维度，避免生成大量 VLM 回答后才发现旧模型无法加载；预检不生成语义或读取 mask。
VLM 缓存阶段每卡各加载一个 FP8 模型、处理不同图像；退出后才启动冻结双塔导出，再退出后训练新头。因此 Qwen 不会与训练反向传播同时占用显存。
最后按类别分到两卡评价，不是把整套评价重复两遍。

如果数据位置不同，设 `SOURCE_ROOT=/.../VisA TARGET_ROOT=/.../mvtec`。
`TARGET_ROOT` 只允许单个目标数据集；多个目标用 `EVAL_DATASETS=mvtec,btad,mpdd`，使用仓库数据根目录配置。
本轮全部 stage 的参数由 prepare 固定；`IMAGE_SIZE/DINO_WEIGHTS/DINO_REPO_DIR/CLIP_PRETRAINED` 等可按现有环境覆盖。

### 先跑一个小规模流程验证

在上面的 `nohup env` 中增加：

```bash
WORK_DIR=./checkpoint/upstream_localization_smoke \
LIMIT_PER_CATEGORY=20 \
HEAD_EPOCHS=2 \
```

这里每个源/目标类别按图像路径哈希选择固定子集，与目标标签和得分无关。
极小子集如果无法给源域正常/异常各保留 train/val，会明确报错，需增大 limit。
这只是检查流程，不是正式结果；正式全量需新目录并恢复 `LIMIT_PER_CATEGORY=0 HEAD_EPOCHS=15`。

### 中断与续跑

相同命令和 `WORK_DIR` 可以重跑。有效 VLM 回答、已导出特征会跳过；head 从完整 epoch 的 `last.pt` 恢复优化器和训练进度。
第一次 seal 后，所有已接受回答（包括无法判断和允许比例内的解析失败）保持不变，防止续跑偷偷改变输入语义。
修改参数、样本、代码、特征或已封存的语义应使用新目录；不能拿旧头配新语义继续训练。

可设 `RUN_REVIEW=0 RUN_EXPORT=0` 复用完整缓存，但保留原命令的其余参数。
`RUN_TRAIN=0` 只跳过训练，评价仍需要 `heads/<mode>/best.pt`。工作目录由 `flock` 锁防止并发破坏。

## 资源与局限

- 全量默认每张图 9 次 VLM 请求；准备日志打印实际总次数。它明显比候选复核慢，首次语义生成可能很久，不能把定位头训练时间当作端到端耗时。
- 保存 Qwen 实际预处理后的两张输入图与原文 trace，用于复查模糊、裁剪和视觉 token 问题。这些 trace 会占磁盘。
- 512 输入时，仅四层 `[4,768,32,32]` FP16 特征约 6 MiB/图，另有 Base 图、语义、压缩开销和视觉 trace。全量建议先预留数十 GB（如 80 GB 余量），实际依压缩率与图片数而变；不是实测容量。
- VLM 看到原图裁剪，但冻结 DINO 仍是 512 输入、16-pixel patch 网格。新头上采样不能凭空恢复所有亚 patch 细节。若小缺陷仍失败，应进一步验证更高分辨率/多尺度 DINO，而不是继续只改提示词。
- 正常预期是模型的一般语义知识，不是真实配对正常参考。它可能不准确，需通过 visual vs vlm 对照检验。
- CLIP 文本超过 77 token 会截断，`sealed.json` 记录截断数量；提示词要求短英文描述。空/不可见语义置零并提供有效性通道，不伪造正常描述。
- 默认一个随机种子是探索性对照，不足以宣称统计显著提升；确认方向后再跑多种子与更多独立目标域。

## 看哪些输出

- `review_audit.json`：各分区请求数、无效数、可用语义数和状态分布。合法“看不清”不算格式错误；整分区无可用语义会停止，不能假装做了 VLM 实验。
- `heads/visual/history.json`、`heads/vlm/history.json`：训练损失、源域验证损失、冻结 Base 源域验证损失。
- `heads/<mode>/best.pt`：仅源域 val loss 选出的头；`last.pt` 用于恢复。
- `results/metrics.csv`、`results/summary.json`：逐类别与宏平均的同批对照。

重点指标：

1. `PRO_exact` / `P_AUROC`：像素/区域排序。`PRO_exact` 是 FPR≤0.3 的精确阈值积分，名称特意与旧 sampled PRO 区分。
2. `I_AUROC`：使用最大像素概率，不混用旧 CLS 全局分类分数。
3. `F1_best`：目标集阈值扫描上界，仅报告，不用于选头或部署阈值。
4. `small_components/small_hits_at_05`：面积≤图像 0.1% 的 GT 连通域中，至少 10% 区域概率≥0.5 的命中数。
5. `recovered_small_at_05/lost_small_at_05`：Base 漏检被恢复/原本命中却退化的小缺陷数。
6. `background_FPR_at_05`、`pixel_recall_at_05`：固定 0.5 阈值下背景误报与前景召回，防止用扩大高分区域换命中率。
7. `changed_fraction`：输出是否真的发生变化。上述 0.5 阈值诊断不是阈值无关性能，需和 PRO/AUC 一起看。

判读建议：

- `visual > base`，但 `vlm ≈ visual`：新头有效，暂未证明 VLM 额外收益。
- `vlm > visual`，且小缺陷恢复增加而误报可控：支持上游局部语义融合方向。
- VLM 头移除语义后几乎不变：可能没有学会利用文字；再检查描述多样性/有效覆盖与融合训练，而不是直接加 MARA。
- 两种新头都下降：优先检查源/目标域差异、监督协议和 DINO 空间分辨率，不归因于 VLM 单一因素。

## 本地验证边界

自动测试覆盖 tile 全覆盖/空间对齐、合法弃权、零初始化、低分区梯度、缓存不可变、GT 隔离、
小型合成特征上的训练/恢复/同批评价和 shell 两卡/DDP 启动参数。语义/特征接口使用模拟对象时会明确标注。
这些是功能与数值测试，不是工业数据上的精度实验；真实 Qwen、双塔权重和两张 4090 的全流程效果仍需服务器实验验证。
