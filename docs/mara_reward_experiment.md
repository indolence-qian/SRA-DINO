# MARA 实验 R：定位导向奖励的受控重训

## 为什么做 R

2026-10-18 收到的 G 报告：288 张选样，原生空 GT 的 BTAD 样本统一排除 1 张，分析 287 张。
四数据集最终类别宏平均 PRO（百分数）：

| 数据集 | Base | G0 阈值 0 | G4 阈值 −0.10 | G5 绕过收益门 |
| --- | ---: | ---: | ---: | ---: |
| VisA | 86.404 | 85.352 | 85.141 | 85.769 |
| MVTec | 80.967 | 80.303 | 79.455 | 79.456 |
| BTAD | 55.284 | 54.972 | 54.635 | 54.635 |
| MPDD | 75.385 | 74.828 | 74.460 | 73.953 |

G0→G4 执行次数 152→321，负奖励执行 5→29，正候选误拒 71→6；四域 PRO 却全部下降。
预测质量收益与真实定位指标并不等价。放宽下界阈值不是当前有效的涨点路线。
旧 no-op 转移的不执行漂移记录 49，修复后 0；本轮 L0/G0 的 PRO 相同。
这证明修复实现正确，但不能证明它已经改善历史训练。
G 不是重训，不能单凭它证明“分类奖励就是唯一根因”。R 是对奖励方向的干预检验。

## 预先声明的三组

| 组 | reward_cls_weight | reward_loc_weight | reward_fp_weight |
| --- | ---: | ---: | ---: |
| R0 重训对照 | 0.5 | 继承模板，当前 1.0 | 继承模板，当前 0.2 |
| R1 弱化分类奖励 | 0.1 | 同 R0 | 同 R0 |
| R2 去掉分类奖励 | 0 | 同 R0 | 同 R0 |

同 seed 三组从完全相同的随机 MARA 参数开始，保存初始参数 SHA256；
旧 MARA checkpoint **只提供冻结 Base 和网络配置**，不加载其中旧策略参数训练。
R0 必须同版本重训，不能拿旧 epoch29 直接当对照。
所有组使用修复后的状态转移，gain_safety_margin=0、gain_accept_threshold=0、
gain_accept_probability=0.5；硬门、预测下界、成本、horizon、Base anchor 均保留。
不使用 VLM、LoRA、AMP、TF32，不修改空间融合门或监督损失系数。

质量定义为

    q = λ_cls·(−CE) + λ_loc·loc + λ_conf·conf − λ_fp·background_probability
    r_t = [q(s_{t+1}) − q(s_t) − c_step − c_refine·1_REFINE]·1_active
    G_t = Σ_k γ^(k−t) r_k

同一个 q 同时生成 GRPO 回报/优势、候选收益回归目标、下界/接受概率监督和 STOP 辅助标签。
不是只改最终显示的 R，更不是把 GRPO 换成 Q-learning。
R2 仍保留全局分类的**监督交叉熵**：只去掉奖励里的分类项，隔离单一因素。
loc 在异常图上为 soft Dice，正常图上为 1−平均异常概率，另有背景概率惩罚。
该代理目标仍不等于 PRO，因此不能保证 R 一定涨点。

直接复用 train_mara.train_one_epoch 的行为采样、三次轨迹重放和全部损失：
0.7 分割 + 0.3 分类 + 0.05 正常图一致性；GRPO 权重 0.5；
gain/gain-consistency 权重各 0.05，STOP 辅助 0.1，稀疏门 0.001，熵 0.01。
默认 30 epoch，前 5 epoch 强制修正热身（Base anchor 除外）、关闭 GRPO 项；
之后正常策略训练。AdamW 1e-4/1e-4，梯度裁剪 1，5% 学习率热身和余弦退火。
单卡独立一组，有效图像 batch=4、组大小=4，全部 R 组相同；这不是旧 DDP 的逐位复刻。
最后一个不足 4 张的 batch 保留，每组样本顺序和 batch 划分一致，不丢样本。

## 数据边界

默认用完整 VisA TEST 有标注池（这是项目原监督来源，不是拿目标域训练），
按类别和真实图像标签确定性 80/20 划分，仅 80% 更新 MARA。
20% 只作最终源域校准。冻结 Base 可能已见过它，所以**不是整个系统的独立验证集**。
不依据校准集挑 epoch，所有组固定最终第 30 轮，目标域仅对该权重评估一次。
目标域 MVTec/BTAD/MPDD 默认沿用 G 的每类 8 张、seed42 元数据配对抽样；
Base、每个 R 组及各个 seed 共用完全相同的目标图和 GT 排除规则。
VisA 此轮评估的是新的源域 20%，不能和 G 的 VisA 96 张数字直接作配对相减。
测试 GT 不输入动作网络，只在整段 GT-free 轨迹返回后评分；训练 GT 仅来自源域训练部分。
原生空异常 GT 统一排除，不修标签；128 网格缺陷消失但高分辨率仍有效的样本保留并标记。

## 双 4090 一体化运行

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main
nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  MARA_CKPT=./checkpoint/mara_visa_hfa3_20260908_225850/ckpt/mara_epoch_29.pth \
  SOURCE_WORK_DIR=./checkpoint/mara_gate_sweep_g_v1 \
  WORK_DIR=./checkpoint/mara_reward_r_v1 \
  GPU_IDS=0,1 TRAIN_SEEDS=0 EPOCHS=30 \
  TRAIN_BATCH_SIZE=4 EXPORT_BATCH_SIZE=8 EVAL_BATCH_SIZE=8 \
  GPU_CACHE_GB=12 CPU_CACHE_GB=4 CPU_THREADS=4 METRIC_WORKERS=2 \
  bash run_exp_mara_reward.sh >> nohup_mara_reward_r.out 2>&1 &
```

SOURCE_WORK_DIR 要指向服务器完整 G 运行目录，不能指向下载的小报告 zip。
没有原目录可删去该行，仍能全流程重新导出。缺省不自动装依赖或下载权重。
先双卡只导出一次冻结 Base，校验复用 G 已有图缓存；新增的源域图只前向一次。
训练不用再加载 CLIP/DINO：两张卡独立并行 R0/R1，随后空出的固定 worker 执行 R2。
不是 DDP，不存在每组再占两张卡的嵌套调度。三个等长任务/两张卡末段有一卡空闲属正常。
多 seed 时预先设 TRAIN_SEEDS="0 1 2" 并使用新 WORK_DIR；完整九组固定分配两卡。

每个 worker 最多保留 4 GiB CPU 缓存、12 GiB GPU 缓存，GPU 预算还受“空闲显存−8 GiB”限制；
LRU 容量不会改变输入/有效 batch/优化目标，已有 cache 是只读硬链接或复制。
不承诺占满显存更快：小 MARA 的核心加速来自不重复前向 Base。
导出阶段可按原规则 OOM 减小推理 batch；训练阶段不偷偷减 batch/group 或跳图。
如训练 OOM，退出并保留完整已提交 epoch。原命令增加 EXEC_GPU_CACHE_GB=0 可关闭 GPU 缓存续跑，
不要修改 GPU_CACHE_GB/训练 batch 等已封存参数；EXEC_GPU_CACHE_GB 只影响执行缓存，不改科学配置。
每 epoch 保存优化器、调度器、随机状态和 loss；同提交同命令同目录可精确恢复保存状态。
不同硬件/CUDA 的浮点逐位一致性不保证。目录锁阻止重复运行，worker 失败打印子日志并停止后续阶段。

```bash
tail -f nohup_mara_reward_r.out
tail -f ./checkpoint/mara_reward_r_v1/train_shard_0.log
tail -f ./checkpoint/mara_reward_r_v1/train_shard_1.log
```

## 看什么结果

上传 WORK_DIR/reward_report.zip 即可，不要打包 evidence/、resume/ 或权重。
里面包含 config/manifest/audit/split/complete 和：

- macro_metrics.csv、metrics.csv：各组与 Base 的类别宏平均和每类别每步结果。
- comparisons.csv：同 seed R1/R2 相对本轮 R0 的配对类别差值。
- gate_summary.csv：实际执行次数、STOP、误拒和负收益执行，正常/异常分开。
- step_trace.csv：完整动作与门控、未加权质量分量、代价和各组回报。
- training.csv：全部 epoch 的监督、GRPO、收益预测、下界和 STOP loss 等。
- checkpoints.json：最终权重 SHA、初始化 SHA、训练预算和缓存/峰值显存。

原始 reward/quality 在 R 组之间定义不同，不能用“R2 的 reward 更大”证明更好；
看统一 PRO/P-AUROC/背景误报，以及未加权 delta_q_loc/delta_q_cls/delta_q_bg。
额外列 reward_reference_R0 将各组实际轨迹事后按统一 R0 系数评分，不参与学习或挑权重。
I_AUROC 延续现有评测的 map-max 图像分数，不等同于 q_cls 使用的 global-logit 分支。
R0 内三份 Base 指标应相同（各 job 单独列出），初始化 SHA 应同 seed 一致。

预期证据链：R1/R2 相对重训 R0 的定位分量改善，随后 PRO/P-AUROC 改善且正常背景误报可控。
若只增执行/代理回报而定位不涨，说明还需研究定位代理目标、空间更新能力或动态新证据，
不能继续把问题归结为门限。seed0 只是机制先导；稳定结论需要至少 3 seed 及完整目标测试集，
不能在目标结果上挑系数/阈值后把同一批目标图声称为独立验证。
