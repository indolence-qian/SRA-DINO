# 小缺陷 VLM 输入与决策诊断

本实验复用已完成的 `run_exp_dual_vlm_local.sh` 导出目录，不训练 Base/MARA，不改原目录，不下载权重。默认抽取 96 个候选，运行四组配对试验及独立的图像依赖检查。

## 为什么诊断

本轮 1,378 个候选中，1,366 个有效回答同时表示证据不足、可见性不足，2 个可见但证据不足，4 个进入增强，6 个解析失败。没有“支持缺陷却被可见性门控拦截”的记录。先检验输入尺度和提示词，不直接放开门控。

`teacher_image_size=512` 是最大尺寸而非固定输入尺寸。裁剪的 `thumbnail` 不放大小图，原最小像素预算只有 1,024。本实验只在 C/D 组提高到 65,536（256²）；插值不会增加原始信息，试验只是检查视觉编码是否浪费已有细节。

## 运行

在原训练环境运行 CPU 准备/评估，在独立 sra_vlm 环境运行 FP8 推理。两卡各一个独立模型进程，按候选分片，非 tensor parallel。每组结束进程退出、释放显存，再开始下一组。需要 Bash 4+、flock 和原两套环境依赖。

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only
conda activate qfg_addino
nohup env \
  SOURCE_WORK_DIR=./checkpoint/dual_vlm_local_visa_v1 \
  WORK_DIR=./checkpoint/vlm_diagnose_visa_v1 \
  TRAIN_PYTHON="$(which python)" \
  VLM_PYTHON=/root/miniconda3/envs/sra_vlm/bin/python \
  VLM_MODEL_ID=/mnt/qfg/Tate_qfg/models/Qwen3-VL-8B-Instruct-FP8 \
  GPU_IDS=0,1 NPROC_PER_NODE=2 \
  DIAG_CANDIDATES=96 DIAG_MIN_PIXELS=65536 DIAG_SANITY_SAMPLES=6 \
  bash run_exp_vlm_diagnose.sh \
  > nohup_vlm_diagnose.out 2>&1 &
tail -f nohup_vlm_diagnose.out
```

`SOURCE_WORK_DIR` 必须替换为上次产生 `candidate_decisions.csv` 的**实验根目录**，不是 `results`、checkpoint 文件或 VLM 权重目录。它应有 `manifest.json`、`config.json`、`local_crops/`、`supports/`、`evidence/`、`evaluation_only/`。默认路径只是此前脚本默认值。旧 `local_reviews` 存在时会选取可见候选作为额外诊断样本，但不会复用旧回答作为新组结果。

## 四组设置

| 组 | 提示词 | 最小视觉像素数 | 最大像素数 |
| --- | --- | ---: | ---: |
| A_original | 原提示词 | 1,024 | 原配置 |
| B_prompt | 中性、分开可见性和异常判断 | 1,024 | 原配置 |
| C_pixels | 原提示词 | 65,536 | 原配置 |
| D_prompt_pixels | 中性提示词 | 65,536 | 原配置 |

候选、裁剪像素、Base、增强幅度、最大上下文、采样参数、正常参考设置全部配对固定。B/D 不强迫缺陷判断，不以“正常参考缺失”自动判定图像不可见。

抽样按类别/候选掩码面积分层，包含旧记录中的可见候选。抽样和推理不打开 GT。该诊断抽样有偏，不用于宣布泛化成绩。`DIAG_CANDIDATES=0` 选择全部候选，仍需独立验证集来评估后续修改。

## 产物与判读

- `selection.csv`：固定候选名单。
- `input_geometry.csv`：原始尺寸元数据、映射框、实际裁剪尺寸及文件哈希。
- `input_audit/*.png`：检测器输入标框＋上下文＋细节。仅人工审查，不送模型；不是原分辨率整图。
- `A_original/` 等四目录：逐候选 JSON，包括原始回答、软件版本、耗时、提示词哈希。
- `*/traces/<候选>/<尝试>/`：`post_vision_*.png`、处理后尺寸、图像网格、token 探测值、停止原因、输出 token 数。
- `sanity/`：正确图、跨类别替换图、灰色空白图的客观描述。独立于异常判定，不能计入检测成绩。默认按较大掩码选 6 个，仍须人工确认图像是否清晰，不能视作已知正确的金标准。
- `results/summary.csv`：四组有效输出、可见率、弃权率、候选召回/精确率/错误增强率。
- `results/paired_transitions.csv`：同一候选跨组的判断变化。
- `results/candidate_audit.csv`：逐候选可见性、理由、预处理尺寸和 token 探测值，与 GT 的诊断对照。
- `results/pixel_effects.csv`：实际修改像素、背景误增强、固定 0.5 阈值的变化和小区域命中。
- `results/area_category_audit.csv`：按类别与候选掩码面积分层的可见性、召回和误增强计数。
- `results/diagnosis.txt`：便于回传的摘要。

token 数来自使用相同预算的 **CPU image processor 探测**，不是 vLLM 内部视觉 token 实测。日志明确区分输入裁剪、qwen_vl_utils 输出和 image processor 网格。没有渲染归一化 tensor 为“实际模型看到的图像”。

判读顺序：原生裁剪是否清楚 → 预处理是否压缩细节 → 客观描述是否随图像变化 → B/C/D 各自改变了什么 → 增强是否准确落在缺陷。低弃权率不是成功标准，必须同时看 GT 正例召回与正常区域误增强。

本脚本不计算抽样子集 PRO/AUC 冒充全量指标，也不自动选最优组或覆盖正式检测配置。未抽中的候选保持 Base，正常判断在默认 enhance-only 模式中不执行抑制。

`pixel_effects.csv` 还包括：所有已选候选直接增强的对照；每组按图像匹配动作数量（严格相同）、掩码面积（贪心近似）的无 VLM 对照；以及仅增强 GT 相交候选的 `GT_overlap_diagnostic_ONLY`。近似匹配须查看实际面积列，不能声称严格同面积。一图只有一个候选时匹配对照可与原动作相同。GT 对照只在离线评估中执行，不进入模型输入、不用于部署，也不是保证涨点的性能上界。

## 缓存与恢复

诊断配置绑定代码、提示词、模型签名、候选名单、源文件内容。源文件、代码或参数改变时拒绝混用缓存，需换 `WORK_DIR`。同一命令重启会跳过完整有效的新组回答；失败/无效候选可重试。保留旧目录，不删除旧实验。运行期间不要同时重跑原导出或修改源缓存。

准备和四组推理不访问 GT，全部组完成后才评估。GT 文件哈希记录在最终摘要中。图像依赖测试的空白/替换图片不进入异常检测指标。

若只有 C/D 恢复可见性，优先修复输入预算；只有 B 改善且错误增强可控，优先修提示词；仍看不清则人工检查原图/裁剪和结构参照。多尺度窗口及正常参考是下一步独立消融，不能和本轮四组同时改动。
