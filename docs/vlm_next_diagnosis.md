# R2 后续：参考输入与局部修正能力诊断

入口：`run_exp_vlm_next.sh`。不重训 Base/MARA/VLM，不修改现有 R0/R1/R2。
需要服务器上的 **完整 R2 目录**，包含 config、manifest、evidence、supports、
evaluation_only、local_crops、local_reviews。只有导出的 CSV 不能运行。

## 一体化命令

在原模型环境中运行（该环境须有 numpy、Pillow、opencv-python、scikit-learn）：

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main

nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  VLM_PYTHON=/root/miniconda3/envs/sra_vlm/bin/python \
  SOURCE_WORK_DIR=./checkpoint/vlm_context_reference_v2/R2_context_reference \
  WORK_DIR=./checkpoint/vlm_next_reference_action_v1 \
  GPU_IDS=0,1 \
  AUDIT_CANDIDATES=192 \
  AUDIT_ALPHAS=0.25 \
  bash run_exp_vlm_next.sh \
  > nohup_vlm_next_reference_action.out 2>&1 &

tail -f nohup_vlm_next_reference_action.out
```

VLM 权重目录默认继承 R2 配置。权重搬家后可显式设置 `VLM_MODEL_ID=/实际/权重目录`。
环境名称/安装位置不同则修改 `VLM_PYTHON`。不需要重新指定 Base checkpoint。
脚本检查解释器、依赖、GPU、输出目录冲突和 R2 数据完整性。
前半段 CPU 评估时 GPU 空闲是正常现象，之后启动两张卡各自独立的 FP8 VLM，
按候选分片，不使用 DDP 或张量并行。脚本不会下载模型或安装环境。

## 两部分实验，分开解释

### P1：完整数据上的离线选择与修正能力

先执行 P1，不启动 VLM，读取 R2 已有决策。使用全部源图像、全部候选支持区域：

| 模式 | 选择依据 | 修改范围 |
| --- | --- | --- |
| base | 无修正 | 无 |
| vlm_cached | R2 缓存 VLM 决策 | 被判为异常的完整候选支持区域 |
| all_candidates | 不看语义，全选 | 全部候选支持区域 |
| gt_candidate_DIAGNOSTIC_ONLY | 候选与 GT 有任意交集 | 被选候选的完整支持区域，仍可能包含背景 |
| gt_pixel_DIAGNOSTIC_ONLY | GT 像素 | 仅候选支持区域内部的 GT 像素 |

所有增强都使用相同固定 log-odds 残差，默认 0.25。不开放抑制。
GT 模式只存在于 `action_eval`，不写入 VLM 决策缓存、不参与选样或提示词。
它们是定位瓶颈的诊断，不是可部署成绩，也不是对每一种指标的严格最优上界。
未进入候选支持区域的 GT 即使在像素级诊断模式也不会被修改。

`AUDIT_ALPHAS=0.1,0.25,0.5` 可在新 WORK_DIR 做预先指定的敏感性检查。
不得根据测试集最优 alpha 宣称无偏部署收益。默认只评估 0.25，避免增加计算成本。

输出 `action_results/`：

- `summary.csv`：类别宏平均指标，包含每个指标有效类别数量。
- `metrics.csv`：每类别、每模式的指标。
- `pixel_effects.csv`：改变的缺陷/背景像素、新增 TP/FP/FN、小/中/大缺陷命中。
- `pixel_summary.csv`：汇总上述计数后计算的像素覆盖率和小缺陷召回率，不平均类别比例。
- `summary.json`：汇总、评估标注文件哈希和指标定义。

**PRO_exact 与旧 PRO 分开报告。** 新诊断使用所有分数阈值、等权 GT 连通域的
加权 ROC 积分，截断至源实验的 max_fpr（一般 0.3）。不使用每个预测图重新划分
的 min/max 阈值网格，也不任意丢弃重复 FPR 点。对比新结果中的 Base 与各模式，
不要把新 PRO_exact 的绝对值与旧 sampled PRO 直接相减。
P-AUROC、图像 max-pixel I-AUROC 与 best-threshold F1 的含义不变。

### P0：同一批候选的三组参考输入审查

从有正常参考且能构造位置打乱对照的候选中，按类别与候选面积分层，
固定 seed 抽取默认 192 个；不足时取全部并报告实际数量。
`AUDIT_CANDIDATES=0` 表示全部可配对候选，不是全部源候选。
选样不读 GT，不按“已知正确/已知异常”筛选。

| 目录 | 输入 |
| --- | --- |
| A_no_reference | 原查询上下文＋原查询细节，无第三张图 |
| B_retrieved_reference | 同一查询输入＋R2 原检索正常参考 |
| C_shuffled_reference | 同一查询输入＋同类别、同一正常源图上另一个位置的参考裁剪 |

打乱参考来自另一个查询候选使用的、不同内容哈希的参考裁剪，优先选候选面积接近的供体。
保留同一正常源图是为了沿用 R2 的“参考源不是当前查询”排除条件，
同时控制物体与光照差异。它是**位置对应关系对照**，不是保证错误或保证正常的标签。
可能偶然仍具有相似结构，必须检查输入图像后解释。

B/C 提示词完全相同，不告诉 VLM 哪个是打乱组。
原检索参考 **不等于已对齐参考**；本轮不引入未经验证的自动配准。
如果两组一样，不能直接下结论“参考无用”，还需核对原参考是否有效、打乱后是否仍可比。
P0 是有参考子集上的诊断，不能代表完整 480 张图的检测性能。

新提示词明确 2/3 张图各自角色，分开返回查询观察、参考观察、局部差异。
解析字段错误仍会重试。已提供参考却声称不存在、可见性与文字冲突等另存语义告警，
不把模型强制改成 matched，也不把警告自动变成增强。
每次请求保存处理后的图片、完整提示词、输出与处理器 token/尺寸探针。
探针不是实际 vLLM 内部每张图的 token 遥测。

输出：

- `selection.csv`、`next_config.json`：源数据、抽样、参考供体、几何信息、代码哈希。
- `A_no_reference/`、`B_retrieved_reference/`、`C_shuffled_reference/`：原始回答与审计。
- 各组 `traces/<候选>/<重试次数>/`：实际预处理图与 `trace.json`。
- `reference_results/summary.csv`：候选级决策、精确率/召回率、匹配与语义告警统计。
- `reference_results/candidate_decisions.csv`：含逐图描述的完整决策。
- `reference_results/paired_transitions.csv`：逐候选的三组动作。

## 如何读结果

1. 若 GT 候选选择优于 VLM，而 GT 像素修正又明显优于 GT 候选：
   同时存在候选判断和候选内定位问题，应分开解决。
2. 若连 GT 像素修正都几乎不改善：检查未覆盖缺陷、残差强度及 Base 分数分布，
   不能仅靠修改 VLM 提示词解决。
3. 若 B 明显优于 C，并能正确描述局部对应关系：参考有条件发挥作用。
   若 B/C 接近且都更容易增强，优先检查第三图引入的判断偏移和参考错位。
4. “没有新增 FN”在仅增强模式下本来就容易成立，还须同时看新增 FP。
   小缺陷为面积不超过图像 0.1% 的 GT 连通域；命中要求至少 10% 的该域像素达到 0.5。

源 VisA pilot 可能与 Base 训练域/训练数据重叠，不能用于声称跨域泛化。

## 续跑与只运行一部分

保持相同配置与 WORK_DIR 重跑，已成功且哈希一致的 VLM 请求会跳过。
改变代码、模型、抽样或 alpha 必须换新 WORK_DIR；不要删除或覆盖旧结果。

- P1 已完成只续跑 P0：原 nohup 命令增加 `RUN_ACTION=0`。
- 只跑 P1：增加 `RUN_REVIEW=0 RUN_REFERENCE_EVAL=0`，不会要求 VLM 环境存在，
  但仍要求完整源 R2 数据和可配对参考，以及原本地模型目录的元数据。
- 只重新汇总 P0：增加 `RUN_ACTION=0 RUN_REVIEW=0`，要求三组缓存已完整。

优先回传两个结果目录内的 CSV/JSON。若参考语义异常，再补充对应候选的三组 traces，
不要只传总指标截图。
