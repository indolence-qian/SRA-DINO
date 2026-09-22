# P0/P1 语义参与诊断

本入口回答：VLM 文本是否实际改变残差、是否传到最终像素、这种改变是否改善定位。
这是推理干预实验，不是新一轮训练，也不把“有变化”直接解释为“语义有效”。

## 运行

在服务器已有 `qfg_addino` 环境中执行。先确认当前 Git 远程指向 SRA-DINO。
本机正确远程名是 `sra`，服务器可直接按仓库 URL 拉取，避免误用指向 AD-DINO 的 origin。

```bash
conda activate qfg_addino
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only https://github.com/indolence-qian/SRA-DINO.git main

nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  REPAIR_WORK_DIR=./checkpoint/upstream_P0_P1_v1 \
  WORK_DIR=./checkpoint/upstream_semantic_audit_v1 \
  GPU_IDS=0,1 \
  AUDIT_BATCH_SIZE=4 \
  AUDIT_CHECKPOINTS=best,last \
  AUDIT_PARTITIONS=val,eval \
  VISUALS_PER_CATEGORY=3 \
  bash run_exp_upstream_semantic_audit.sh \
  > nohup_upstream_semantic_audit_v1.out 2>&1 &

tail -f nohup_upstream_semantic_audit_v1.out
```

需要完整 P0/P1 目录中的配置、两个头的最佳检查点和 VLM 最后检查点。
原始 `upstream_localization_v1` 缓存路径从 `repair_config.json` 自动读取，
其中封存特征、配置、manifest，以及原始 RGB/mask 必须存在。
仅有上次结果 zip 无法运行，因为 zip 不含原始特征缓存。
不需要重新生成缓存、下载权重或启动 Qwen/vLLM。需要训练环境中的 matplotlib。

两张卡按类别独立推理，不是 DDP 训练。CPU 执行精确 PRO，可能是主要耗时。
每完成一个类别/检查点保存一次。中断后原命令续跑，已完成项跳过。
修改诊断设置或检查点时使用新 WORK_DIR。所有旧实验文件只读，原有哈希保持有效。

## 对照定义

所有语义对照固定同一个 VLM 头、视觉特征、Base、图像和 alpha：

| arm | 干预 |
|---|---|
| base | 原始 Base 概率 |
| real | 原始局部文本 embedding |
| zero_text | embedding 置零，但保留 valid、boxes 和覆盖信息 |
| branch_off | 使用原实现的 use_semantics=False，同时移除文本和覆盖信息 |
| shuffled_text | 使用同分区、同类别其他图像的文本，保持查询图的 valid、boxes 和文本角色 |

错配优先使用相同 tile 位置的有效文本，无可用文本时使用其他 tile 的同角色文本。
donor 选择只依赖有效性、图像 ID 和固定 seed，不看标签、缺陷面积、Base 分数。
无其他图像有效 donor 时原 embedding 保留，明确计入 unavailable_slots。
即使 donor 不同，其文本也可能相同，必须核对 changed_slots，不能将无效错配解释成语义无效。
原语义本身无效的位置不被补成有效，以免将覆盖差异误当成内容差异。

默认比较 best 和 last，两者都使用 **best 在源验证集选出的 alpha**。
last 仅作诊断，不根据本轮目标指标重新选 checkpoint 或 alpha。
如果 best 的 alpha=0，最终概率相同是预期行为，raw residual 仍单独记录。
另外用纯视觉头验证“改变文本不改变视觉头输出”，并检查 alpha=0 精确还原 Base。

## 输出和判读

- `README.md`：参与数量与宏平均 PRO 对照表。
- `participation.csv`：分区/检查点/干预的有效文本、有效错配、残差响应、像素响应、相对 Base 修改图像数。
- `images.csv`：每图每种干预的输入有效性、raw residual 幅度、相对真实语义和 Base 的变化均值/最大值/像素比例。
- `metrics.csv`：每类别及宏平均 PRO_exact、像素/图像 AUROC、固定 FPR 下区域/小缺陷召回等。
- `comparisons.csv`：同检查点的 real 减去每个对照的指标差；CSV 中差值单位为比例，README 的 PRO 差值为百分点。
- `components.csv`：每个 GT 连通域的面积、Base 低分比例、修正前后召回、固定 0.5 下恢复/丢失。
- `regions.csv`：小缺陷、大缺陷、邻近背景、远处背景、Base<1e-5 缺陷像素的计数和分数/残差总和。均值需总和除以像素数，空区域不作为零分样本。
- `donors.csv`：每个有效错配的图像、tile、角色与 donor，可追溯且与 batch size 无关。
- `panels/`：原图、GT、Base、真实/零/错配语义输出、真实减 Base、真实减错配。概率共用 log 色标，差分共用对称线性色标。
- `groups/`：断点文件，含分支 forward 调用计数、负对照检查结果及原始明细。
- `summary.json`：汇总和配置/权重/代码指纹。

默认每类别每检查点画 3 张，图像按 ID+seed 选取，不用 GT 或成绩挑选好看的样本。
每图的 image_id 可在原缓存 manifest 中定位；panel 文件名为分区类别和 image_id 的摘要。
小缺陷沿用面积≤图像 0.1%、命中要求区域至少 10% 像素过阈值的定义。
邻近背景默认是 GT 外 8 像素方形膨胀带，`NEAR_BG_RADIUS` 可覆盖。
低分 GT 与大小缺陷分组有重叠，不能把所有 region 的计数直接相加。

### 四步证据

1. 有效输入：检查 valid_slots、real_embedding_abs_mean，以及实际 changed_slots。
2. 残差参与：改变内容后 raw_vs_real 是否变化。模块被调用但 raw 不变，不足以证明内容参与。
3. 像素参与：prob_vs_real 是否变化。raw 改变但概率几乎不动，需要看 alpha 和 Base 的极低分。
4. 有益参与：real 是否在同检查点下优于 shuffled/zero，尤其是源验证集的小缺陷和背景指标。

`real` 的“相对 real 变化”必然为零；应看它的 `prob_vs_base` 判断整体定位头是否修改输出。
仅 branch_off 敏感、zero_text/shuffled_text 不敏感，可能说明模型依赖覆盖或分支偏置。
推理置零/错配会改变输入分布，因此这些对照不能替代从头训练的严格语义消融。
单一错配 seed 和单轮结果不证明统计显著收益；本轮先验证信息通路和修正位置。

val 和 eval 分开报告，所有标签仅用于离线统计、阈值诊断和图示。
FPR 阈值不是部署阈值，不用目标标签训练或选参。
本次不修改损失或加入空间匹配新头，避免在未确认语义通路前同时改变多个因素。

## 本地验证

```bash
python -m unittest discover -s tests -p 'test_upstream_semantic_audit.py' -v
```

测试覆盖已知语义敏感/不敏感头、内容置零与覆盖区分、错配 donor 和缺失处理、
残差有变化但 alpha=0 无像素变化、旧缓存逐文件不变、断点续跑、输入变更拒绝、
小缺陷/邻近背景统计及双 GPU shell 分片和失败停止。合成测试不代表真实精度收益。
