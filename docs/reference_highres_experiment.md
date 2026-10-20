# 一阶段正常参考与高分辨率四组实验

本实验借鉴 IAD-Unify 的正常参考密集比较和 GenAU 的全局/局部高分辨率读取，
验证二者能否在已有 CLIP+DINO Base 上产生独立收益和交互收益。只训练新的一阶段小头，
不训练原 Base，不运行 Qwen/VLM 或 MARA。不是这两篇论文的复现，也不预先声明涨点。

## 实验矩阵

| 结果 | 高分辨率 | 正常参考 | 用途 |
|---|---|---|---|
| base_global | 否 | 否 | 冻结原 Base；preflight 与旧 forward 校验 |
| base_highres | 是 | 否 | 全局图和原图切块的固定融合，无新头 |
| A_global | 否 | 否 | 匹配结构的纯全局新头 |
| B_highres | 是 | 否 | 高分辨率新头 |
| C_reference | 否 | 是 | 全局正常参考证据新头 |
| D_highres_reference | 是 | 是 | 两种证据结合 |

四个头共享结构、初始化种子、每轮样本顺序、有效 batch、更新步数、优化器、损失与源验证规则，
各自独立训练。不存在的参考输入全部置零并提供有效性平面；A/C 的低分辨率特征插值到相同头网格，
不因此获得新的视觉信息。B/D 以固定 `HIGH_WEIGHT=0.25` 融合两路特征、先验和冻结概率图。
每个头的零残差严格恢复其自身冻结读出（A/C 为 global，B/D 为 highres）。

## 输入和学习路径

- 全局图：原图缩到 512，DINO 的四层 Patch 网格为 32x32。
- 局部图：在**原图缩放前**做全覆盖重叠裁剪，默认 2x2、边长占比 0.6，每块再缩到 512；
  用正值 Hann 权重按原坐标拼到 64x64 特征网格，减弱块边缘接缝。不是只对 Base 高分 ROI 裁剪。
- 参考：每类别从已知正常 TRAIN 图中按确定性哈希建立 4 图池，按查询全局 DINO 描述子选 1 张；
  全局 Patch 匹配全局参考，局部 Patch 匹配参考图的所有局部 Patch。
- 对应：在 HFA 后、逐图 Patch 标准化前的原始 DINO L2 特征上，分块计算余弦匹配；
  默认半径 0.35、坐标距离惩罚 0.15，top-3 soft matching。缓存带符号残差、距离、top-k 熵。
  `MATCH_RADIUS=0` 允许全局搜索，可用于位置约束消融；过小半径导致候选不足会明确失败。
- 缓存：同一固定正交随机投影将查询/残差从 1024 压到 64 维，各层保持空间结构；
  查询与残差可重构压缩的匹配参考。随机投影只是节省缓存，**没有文本对齐含义**。
- 新头：读取查询、匹配参考、带符号/绝对残差、距离、熵、坐标、原跨模态先验和有效性；
  学习投影到冻结 CLIP 正常/异常 Prompt Adapter 的特征空间，再结合余弦 margin 预测 Base log-odds 残差。
- 损失：各组同样使用 BCE+正例 Dice、背景抬升约束、残差约束，以及语义匹配辅助 BCE。
  `SEMANTIC_WEIGHT=0` 可用于辅助对齐消融，需要新 WORK_DIR。

当前固定 CLIP。dino.txt 配套视觉 head/文本塔替换是下一步独立对照；本入口不把 dino.txt 文本直接套到原四层上，
也不声称 DINO 残差天然与任何文本空间对齐。建议先判明高分辨率/参考的收益，再扩展文本塔。

## 数据与评价协议

源数据默认 VisA TEST（沿用已有有监督 Base 协议），按类别/正常异常分层，以解码 RGB 内容家族切为 head train/val。
原 Base 可能见过源 val，因此源 val 仅用于新头选参，不能当独立泛化成绩。
目标为 MVTec/BTAD/MPDD TEST，mask 不参与特征导出、检索、训练或选模。

C/D 使用同类别的目标正常 TRAIN 图，因此是 **normal-reference assisted / one-reference** 设置，
不能把其分数直接宣称为比 A/B 更好的严格无参考零样本方法。它们的差值回答的是额外正常支持的价值。
所有参考池排除全部查询的解码 RGB 重复内容，包括重新编码的图片；不从目标 TEST 选择已知正常参考。

只用 source val 的 BCE+正例 Dice 选择 epoch 和 alpha（0/0.25/0.5/1），显式保留 alpha=0 回退。
epoch=0、alpha=0 表示冻结读出回退，不能算新头学习收益。
背景分位数阈值同样只来自 source val，并跨目标数据集复用；不保证目标 FPR 相同。
掩码按最近邻缩放，VisA >0，其余 >127。它与旧双线性掩码协议不同，应看本轮同批差值。

输出像素 AUROC、AP、精确 AUPRO-30、图像 max-pixel AUROC、F1_best，以及面积不超过 0.1% 的小连通域命中/
恢复/丢失和背景误报。F1_best 是仅供报告的目标阈值上界。
默认评价 512，可用 `EVAL_SIZE=1024` 对各组统一放大输出并使用更细 GT；它不会再增加视觉信息，不能称原生 1024 输出。

## 一体化运行（服务器）

使用原来的训练环境与权重，不需要 VLM 环境。每张卡一个独立 worker：分片提取证据，
随后每卡训练一个独立头，四组完成后再分片评价所有目标类别。

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
conda activate qfg_addino

nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  BASE_CKPT=/mnt/qfg/Tate_qfg/SRA-DINO/checkpoint/base_visa_hfa3_20260908_225850/ckpt \
  WORK_DIR=./checkpoint/reference_highres_v1 \
  GPU_IDS=0,1 \
  SOURCE_DATASET=visa \
  EVAL_DATASETS=mvtec,btad,mpdd \
  HEAD_EPOCHS=15 TRAIN_UNTIL_EPOCH=5 \
  HEAD_BATCH_SIZE=8 MICRO_BATCH_SIZE=2 \
  bash run_exp_reference_highres.sh \
  > nohup_reference_highres_v1.out 2>&1 &

tail -f nohup_reference_highres_v1.out
```

首次先跑 5 轮筛选，后续相同配置和 WORK_DIR，设置 `TRAIN_UNTIL_EPOCH=15` 即可续跑；
`HEAD_EPOCHS=15` 始终不变以保持学习率计划。已完成且哈希有效的导出会跳过。
变更参考、分辨率、损失、特征压缩、代码或权重，需要新 WORK_DIR；原实验不会被覆盖。

首次流程验证可设 `LIMIT_PER_CATEGORY=20 HEAD_EPOCHS=2 TRAIN_UNTIL_EPOCH=2`，并用新目录；
小子集必须包含足够正常/异常样本。正式恢复 `LIMIT_PER_CATEGORY=0`。
只评估已有训练可设 `RUN_EXPORT=0 RUN_TRAIN=0`；仍会验证封存缓存。

数据根目录沿用 `Datasets/__init__.py`，也可设置 `SOURCE_ROOT`、单目标时 `TARGET_ROOT`，
或 `DATA_ROOTS_JSON='{"visa":"/data/VisA","mvtec":"/data/mvtec","btad":"/data/BTAD","mpdd":"/data/MPDD"}'`。
需要匹配 Base 的 `HFA_SETTING/DINO_WEIGHTS/CLIP_PRETRAINED`。默认目录选择原有数字 epoch，
拒绝 MARA/单塔检查点；非数字命名的 Base 请传具体文件。若 Base 内嵌模型路径，须显式传对应路径。

## 结果与消融顺序

- `results/README.md`：六种结果的类别宏平均。
- `results/summary.csv`、`metrics.csv`：宏平均与逐类别完整指标。
- `results/comparisons.csv`：新头收益、高分辨率收益、参考收益，以及 `D-B-C+A` 交互，单位百分点。
- `source_screen.csv`、`heads/<arm>/history.json`：实际完成轮次、源选择的 epoch/alpha/阈值、损失、吞吐。
- `features/*.json`：具体参考 ID、相似度、原图尺寸、裁剪坐标、查询提取耗时与峰值显存。

导出同时计算所有分支，query_seconds 不等于各 arm 单独部署时延；日志 total_seconds 还包含参考池编码。
GPU 峰值也不是严格的各 arm 模型比较。实际部署速度须另测，不能把小头时间当端到端时间。
缓存启动时估算体积；FP16 特征、FP32 概率、未压缩 NPZ，完整多目标数据可能需要数十至上百 GiB，
请以日志和实际目录大小为准。减小 FEATURE_DIM 可省空间，但属于额外消融。

先看 B-A、C-A、D-B、D-C；再以新目录比较 REFERENCE_COUNT=1/4、MATCH_RADIUS=0/.35、
SEMANTIC_WEIGHT=0/.1、FEATURE_DIM=64/128、多种子和独立目标。MVTec 已多次用于项目诊断时，
最终论文还需要未用于方案选择的独立目标验证。

本地合成测试覆盖裁剪覆盖/接缝、对应坐标、同图排除、四组输入隔离、语义与残差梯度、
不读取目标 mask 的导出/训练、缓存变更失败、断点恢复和脚本阶段/失败停止。
它们不能替代服务器真实 DINO/CLIP 权重的 preflight 与正式跨数据集精度实验。
