# VLM 修复 v2：先离线复核，再验证上下文和正常参考

## 已确认的问题与修复边界

B/D 最后一次原始回答的 99 条解析失败，全部来自空缺陷类型和/或超过 400 字符的解释。修复只处理非决策元数据：不确定/正常回答的空字符串类型规范为 unknown；超过 400 字符的解释保留原文并记警告。仍拒绝非 JSON、重复键、字段缺失/新增、候选编号错误、非法枚举、虚构参考关系、空理由，以及支持缺陷却没有缺陷类型的回答。不会因为解析放宽而把不确定改成缺陷。

旧 `legacy` 解析与旧默认脚本行为保留。新实验显式使用 `LOCAL_PARSER=repair_v2`，把策略和代码哈希写入配置。修复版的可见性提示要求字段与理由一致，但不能保证模型自报可见性客观正确，仍需人工看图。

## 第一步：无需 GPU 的旧回答重评

```bash
conda activate qfg_addino
nohup env MODE=reparse \
  SOURCE_DIAG_DIR=./checkpoint/vlm_diagnose_visa_20260912 \
  WORK_DIR=./checkpoint/vlm_diagnose_visa_repaired_v2 \
  TRAIN_PYTHON="$(which python)" \
  bash run_exp_vlm_repair.sh > nohup_vlm_reparse.out 2>&1 &
```

源目录是旧四组诊断目录，需包含 diagnosis_config.json、A/B/C/D 逐候选 JSON 和 traces，并且其配置指向的原 Base 导出目录仍可访问。不是只传来的 B.zip/D.zip。本地压缩包只能核对回答，完整像素重评需要服务器预测缓存。

只重放每个候选最后一次回答，不在多次尝试中挑最有利答案。不调用 VLM、不加载 Base 权重、不训练。验证旧数据哈希后，在独立目录保存新配置、新判断、trace 副本和 `reparse_changes.csv`，再生成 `results/`。源文件不变；新结果仍是原诊断子集，非全量 PRO/AUC。预期这批数据 B 的支持缺陷数 6→8、D 的 9→10，不承诺指标上涨。

## 第二步：双卡观察范围/参考消融

```bash
nohup env MODE=ablation \
  BASE_CKPT=/mnt/qfg/Tate_qfg/SRA-DINO/checkpoint/base_visa_hfa3_20260908_225850/ckpt/14.pth \
  WORK_DIR=./checkpoint/vlm_context_reference_v2 \
  TRAIN_PYTHON="$(which python)" \
  VLM_PYTHON=/root/miniconda3/envs/sra_vlm/bin/python \
  VLM_MODEL_ID=/mnt/qfg/Tate_qfg/models/Qwen3-VL-8B-Instruct-FP8 \
  GPU_IDS=0,1 NPROC_PER_NODE=2 DATASETS=visa MAX_PER_CATEGORY=40 SEED=42 \
  bash run_exp_vlm_repair.sh > nohup_vlm_context_reference.out 2>&1 &
```

顺序运行 R0（修复解析/提示词、原上下文）、R1（仅扩大上下文，4→8 倍包围框，最小边 32→64 检测图像素）、R2（与 R1 相同，增加正常训练图参考）。细节裁剪、候选提议、修改掩码、增强幅度和最小视觉像素预算保持固定。每组重新导出冻结模型以形成独立配置，不重训。不要和其他占用两张卡的实验同时运行。

正常参考仅来自训练集已知正常图片，排除原图路径与重复内容；原有外观距离过滤上增加梯度方向余弦过滤。它只是保守匹配预筛，不是特征级配准，也不保证位置/语义相同。VLM 必须判断参考是否可比；没有合格参考则不给参考，不能拿测试正常图补齐。R2 是否真的提供了参考，应从候选记录的 reference 字段/has_reference 和 reference_match 统计确认，而不是仅凭开关。

默认禁止抑制，修改只发生在原掩码内。本轮不同时引入新的像素修正算法，避免和提示词/上下文/参考混淆。先验证这两项是否提高缺陷候选召回且不过度误增强，再决定是否引入边界/特征约束的像素修正。

结果：各组 `results/metrics.csv`、`diagnostics.csv`、`candidate_decisions.csv`；原始 local_reviews JSON 记录 `parser_policy`、`parse_errors`、`parse_warnings`。不确定仍是不确定，警告不算解析失败。改代码/参数需新输出目录，不覆写旧实验。
