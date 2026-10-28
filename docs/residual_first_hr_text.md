# 残差优先：正常参考、高分辨率与文本差异对齐

入口：`run_exp_residual_text.sh` / `residual_text_experiment.py`。
这是独立的新实验，不读取或改写旧 Base、reference_highres_v1、alignment_three_stage_v2 的权重/缓存。
不运行 RL、VLM 或 MARA。不要把新入口指向旧 WORK_DIR。

## 1. 实际实现的网络

```
查询原图 + K 张同类别正常 TRAIN 图
  └─相同的全局512 + 原图2×2重叠裁剪（每块宽高占比0.6，输入512）
      └─共享、冻结的原始预训练 DINOv3-L/16，层5/11/17/23
          └─原生 LayerNorm + 每个 token 的 L2 归一化
              └─同层、同尺度正常池的精确最近邻（全部 K 张，非4选1）
                  └─1024维相减 -> 固定随机投影256维 -> 源训练正常残差RMS缩放
                      ├─浅层残差约束器 -> 可逆 affine coupling Flow -> 异常分数
                      └─无偏置 MLP -> 冻结 CLIP 文本状态差 -> 语义分数
                          └─逐视图分数融合 -> 原坐标 Hann 拼接 -> global/local=.75/.25
```

查询/正常参考都经过 DINO。全局特征只匹配正常全局池；局部特征匹配所有正常局部块。
最近邻在完整1024维空间计算，不先压缩；残差保留符号与幅度，不再做单位长度归一化。
不使用逐图均值方差标准化、原图特征直通、旧 Base 概率、原 HFA、Patch/Prompt Adapter。
缓存只允许 residual、distance、match_index、boxes、pixel_boxes；不保存查询外观特征供头读取。
match_index 对应参考元数据的图像/视图/patch 拼接次序，可用于检查错误对应。

这是 **ResAD-inspired** 方法，不是原论文逐项复现：骨干换为 DINOv3，约束器使用无偏置、
同维、带恒等跳连的浅 MLP，保留负残差并从恒等映射开始，不采用原版 Conv+BN+ReLU。
Flow 为真正可逆的8层 affine coupling，含固定置换及 log-Jacobian 项；不需要 FrEIA 新依赖。
损失为正常 NLL、正常紧凑约束、异常残差保持和正常边界之外的异常分离，边界由源批次正常NLL给出。
这些实现差异必须在论文中说明，不能直接将结果称为原版 ResAD。

文本投影器为每层独立 `256 -> 256 -> text_dim` 无偏置 MLP，全局/局部共享；零残差必输出零语义证据。
文本特征使用同一模板下的 `normalize(T(异常状态))-normalize(T(正常状态))`，多模板平均后归一化。
CLIP 来自显式指定的本地 OpenAI 格式 `.pt`/state_dict，不自动下载；不执行 CLIP 图像前向。
text_dim 从实际权重推断；不假设 DINO 残差与 CLIP/dino.txt 原生对齐。
文本模块读取约束前的残差，不能读取 Flow 高斯潜变量或标量异常图。
文本分数的正常中心取 `max(0,源正常95%分位数)`，确保零残差经过校准后也不会产生正向文本增量。
无位置约束的最近邻不保证结构对应，缺件/错位可能出现假正常匹配；需要查看match_index，本版不声称解决逻辑异常。

## 2. 数据及评价边界

- 源域默认 VisA TEST（正常/异常及其mask）作为辅助有监督数据；正常参考来自正常 TRAIN。
- 源域按**类别留出**验证，默认25%类别，内容重复不能跨训练/验证/目标分区。
- 目标默认 MVTec/BTAD/MPDD TEST；每类别固定 K=4 正常 TRAIN 支持，排除全部查询的RGB解码重复。
- K=1/2/4 均可运行，但需分别使用新目录；每次所有组用同一 K、同一支持集。
- 正常参考图可在未见类别推理时编码，不做任何目标梯度更新或 BN 统计更新。
- 属于 **K-shot normal-reference assisted / cross-category no-retraining**，不是无参考零样本。
- 正常残差RMS、分数校准只用源 TRAIN 的完全正常单元；不对每张测试图做 min-max。
- 小缺陷从原生mask直接池化为覆盖率，覆盖率>0的混合单元不当作正常单元；不先缩小mask再池化。
- 模型/融合强度按源 VAL 类别宏平均 PRO、AP 选择，默认选择网格256；最终评价512。
- 操作阈值在最终评价分辨率上用源 VAL 正常像素的99%分位数校准，不保证目标FPR也为1%。
- 所有目标预测先写入并锁定哈希，之后 evaluate 才打开目标mask。配置不读取目标mask内容或类型词。
- MVTec、BTAD、MPDD 已参与此前方法开发；最终论文仍应另设未参与设计的目标基准。

高分辨率需要原图真实包含细节。输出插值到512/1024不等于原生逐像素检测；单查询需5次DINO前向，
参考池首次还需5K次编码，可按类别复用。缓存导出包含全部分支的耗时，不是单arm部署时延。

## 3. 一体化运行（Linux服务器）

沿用现有训练环境，不需要安装新的Flow库。必须有本地 DINO 预训练权重、CLIP 原始预训练权重和数据。
**不再需要 BASE_CKPT 或 V1_DIR。** 检查 CLIP_WEIGHTS 是真实文件，而不是 `openai` 字符串。

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main
conda activate qfg_addino

nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  DINO_WEIGHTS=./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  CLIP_WEIGHTS=./CLIP/ckpt/ViT-L-14-336px.pt \
  WORK_DIR=./checkpoint/residual_first_hr_text_v1 \
  GPU_IDS=0,1 TRAIN_SEEDS=42 REFERENCE_COUNT=4 \
  EPOCHS=15 TRAIN_UNTIL_EPOCH=5 \
  bash run_exp_residual_text.sh \
  > nohup_residual_first_hr_text_v1.out 2>&1 &

tail -f nohup_residual_first_hr_text_v1.out
```

数据路径沿用 `Datasets/__init__.py`；可设置 SOURCE_ROOT，或：

```bash
export DATA_ROOTS_JSON='{"visa":"/data/VisA","mvtec":"/data/mvtec","btad":"/data/BTAD/BTech_Dataset_transformed","mpdd":"/data/MPDD"}'
```

单卡设置 `GPU_IDS=0` 即可，G/HR顺序训练。两卡分别处理G/HR，各用一个worker，不是DDP。
更多卡仅用于分片导出/预测/评价；只有两个独立模型训练任务。

第一次建议新目录小规模真权重冒烟：

```bash
WORK_DIR=./checkpoint/residual_first_smoke \
LIMIT_PER_CATEGORY=20 EPOCHS=2 TRAIN_UNTIL_EPOCH=1 \
GPU_IDS=0,1 bash run_exp_residual_text.sh
```

小样本截取是固定哈希、不是按目标标签挑样本；如源训练/验证缺少正常或异常图，增加 LIMIT_PER_CATEGORY。
小样本不能作为正式精度结果。正式实验必须新目录、`LIMIT_PER_CATEGORY=0`。

### 续跑与恢复

主目录先跑5轮后，保持所有计划参数不变，将 `TRAIN_UNTIL_EPOCH=15` 后重跑同一命令。
**EPOCHS始终保留15**，以保持学习率调度。导出、文本编码和已完成阶段会验证后跳过。
through5、through15 各自锁定模型/预测/报告；后续训练不覆盖旧的through5快照。
中断恢复使用last.pt中的优化器、调度器和确定性逐epoch数据/采样顺序。

只重新评价现有快照可以设置 `RUN_EXPORT=0 RUN_TRAIN=0 RUN_PREDICT=0`，保持原GPU数量与配置。
只做源阶段设置 `RUN_PREDICT=0 RUN_EVAL=0`。工作目录有flock独占锁；worker失败不启动后续阶段。

改变代码、权重、参考数量、压缩维度、原始分辨率、数据划分、语义标注或模型结构都要求新 WORK_DIR。
不要把此前异常导出修复用的 REPAIR_EXPORT / START_PHASE 参数传给本入口；新管线没有旧Base parity重算。

### 显存和存储

默认 `FEATURE_DIM=256 BATCH_SIZE=8 MICRO_BATCH_SIZE=1 TOKENS_PER_VIEW=256`。
有效batch通过梯度累积实现；density训练每视图分别采正常/异常token，避免漏掉小异常。
只有DINO编码可选BF16，Flow和文本投影训练使用FP32以稳定似然计算。
若DINO导出显存不足，先减小 MATCH_CHUNK/BANK_CHUNK（更改配置需新目录）；若训练不足，降低batch或micro-batch。
不要在一张卡上同时启动两份手工训练进程。

默认单图残差约10 MiB，8张最终分数图约8 MiB，总量可能超过100 GiB。
prepare会打印缓存估算；预留查询缓存、预测图及临时文件空间，禁止直接复用旧64维缓存冒充新特征。

## 4. 默认文本能力与可选细粒度标注

默认仅用一对通用正常/异常状态，使用源二值mask训练残差投影器：BCE+Dice、正常紧凑及异常幅度约束。
这只验证通用异常方向接口，**不能宣称模型理解裂纹/污染/缺件或未见缺陷词**。
文本编码器和提示词固定，不实施可学习上下文；共享可学习提示词留作后续独立消融。

如确有源域区域语义监督，可通过 `SEMANTIC_ANNOTATIONS=/absolute/path/source_regions.json` 启用类型方向。
JSON格式如下（示例路径需替换成当前源域真实文件，不能指向目标）：

```json
{
  "version": 1,
  "types": [
    {"id": "crack", "normal": "an intact surface", "abnormal": "a cracked surface"},
    {"id": "contamination", "normal": "a clean surface", "abnormal": "a contaminated surface"}
  ],
  "regions": [
    {"image_path": "/data/VisA/source_image1.JPG", "mask_path": "/data/source_regions/crack1.png", "type_id": "crack"},
    {"image_path": "/data/VisA/source_image2.JPG", "mask_path": "/data/source_regions/contamination2.png", "type_id": "contamination"}
  ]
}
```

区域mask需与原图同尺寸，0/255编码，非空且完全包含于该图原始异常mask；每个类型至少有一个源训练区域。
类型对比至少需要两个类型。重叠合法类型视为多正样本，不作为相互负样本；不使用整图标签覆盖全部patch。
标注指向目标图、正常区域或仅存在于源验证类的类型都会报错。
有标注时预测额外保存 `types_G/types_HR`，类型顺序对应 prompts.json 中generic之后的顺序；它们是匹配分数而非概率。
类型词只输出解释图；主检测融合仍使用通用异常方向，不靠增加词表扩大max分数。

## 5. 输出、对照与回退

8组共用同一支持集：

| 组 | 内容 |
|---|---|
| D_G / D_HR | 最近邻距离，全局 / 全局局部融合 |
| R_G / R_HR | 源验证选出的残差分布检测（不合格则回退距离） |
| T_G / T_HR | 独立文本残差分支 |
| F_G / F_HR | 残差检测 + 源验证选出的非负文本增量 |

Flow候选必须同时不低于距离基线的源PRO/AP且至少一项提高，否则 `density_kind=distance, density_epoch=0`。
文本投影器按源PRO/AP选epoch；`text_epoch=0`表示初始化胜出而不是训练成功，此时禁止非零融合。
融合候选lambda=0/.05/.1/.25/.5；不同时满足源PRO/AP不退化条件就选0。
最终 `S=R+lambda*relu(T_calibrated)`，文本不否决未描述到的异常；仍可能抬高误报，必须同时查看FPR。
回退/未训练/零融合在报告中明确标记，不能当作新模块改善。

结果位置：

```
WORK_DIR/
  config.json / manifest.json / references.json / prompts.json
  text_bank.npz / normal_stats.json
  features/train|val|eval/<id>.npz          # 逐视图残差及匹配索引
  sealed_source.json / sealed_eval.json
  heads/seed42/density_G|density_HR|text_G|text_HR/
    history.json / last.pt / best.pt / through5/best.pt
  decisions/seed42/through5/
    selection.json / interface_audit.json
  predictions/seed42/through5/
    <id>.npz / locked.json
  reports/through5/
    README.md / summary.csv / per_seed.csv / metrics.csv
    comparisons.csv / source_selection.csv
  panels/seed42/through5/<dataset>/<category>/*.png
```

先看 summary.csv（类别宏平均）和 comparisons.csv（R-D、HR-G、F-R百分点），
再看 source_selection.csv 判断是否真实选中训练后的Flow/文本。逐类别完整指标在metrics.csv。
TRAIN_SEEDS=42,43,44 时另报训练种子标准差，不混成像素独立样本置信区间。
图像分数取最终图最大值；small指标统计面积<=0.1%的连通域在源阈值下至少10%像素命中。
F1_best是只供评价的目标阈值上界，不参与部署或选模。

panel为每类固定哈希选出的两张图，显示query、GT及五种分数；色彩采用固定sigmoid显示，不逐图min-max。
`interface_audit.json` 检查零残差语义为零、文本方向翻转能改变实际评分；这只是接口检查，不是语义理解证明。

## 6. 本地验证与限制

```powershell
$env:SRA_TEST_BASH='D:/qfg/编译器/Git/bin/bash.exe'
& 'D:/qfg/python/Scripts/python.exe' -m unittest tests.test_residual_text -v
```

测试使用显式的测试替身，不提供生产用合成特征开关。覆盖最近邻与tie、Flow逆变换/Jacobian、
梯度、零残差、文本方向依赖、多正样本、原生小缺陷覆盖、Hann晚融合、类留出/重复排除、
源区域语义校验、缓存篡改、目标mask隔离、完整CPU冒烟、续训快照和Bash单双卡编排/失败停止。
本地没有完整预训练权重、CUDA/BF16或服务器数据；本地通过不等于正式精度、GPU显存和耗时已经验证。

方法参考：[ResAD](https://arxiv.org/abs/2410.20047)、[Res²CLIP](https://arxiv.org/abs/2605.16171)。
本入口是两者思想与当前高分辨率管线的受控组合，不因更换DINO或引入文本差就自动形成论文创新。
