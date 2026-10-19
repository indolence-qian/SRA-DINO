# MARA：冻结环境下的 PPO + V 改造

## 本轮要回答什么

把旧 MARA 的 GRPO 风格策略训练改成显式维护状态价值的 on-policy Actor-Critic。
这不是 Q-learning，也不是“旧代码完全没有多步回报”。旧版已经使用 return-to-go；
这次新增的是独立的 V 网络、严格保存行为状态的 PPO 更新，以及统一的无收益否决门环境。

原版 `train_mara.py` 和 G/R 入口保留，不覆盖历史实验。新入口是
`mara_ppo_experiment.py`，核心是 `tools/mara_ppo.py`。

默认两组、每 seed 同参数初始化：

| 组 | 策略优势 | V 网络训练 | 其他条件 |
| --- | --- | --- | --- |
| V | `G - V_old(s)` | 完整 MC 回报回归 | 全部相同 |
| MC | `G`，零基线 | 不更新，V 恒为 0 | 全部相同 |

这组比较隔离 V 基线的影响；不是把新 PPO 与历史 R0 的差异全部归因于 V。
相对旧 R，还改变了冻结 Refiner、HR 奖励、STOP 成本、收益否决门等条件。
首次 seed=0 是机制试跑；稳定结论需预先声明至少 3 个 seed 并完整报告，不选最优 seed。

## MDP、网络与优化

Base 的 CLIP/DINO、证据库、以及**已经训练过的** Refiner 全部冻结。
`MARA_CKPT` 中的旧 Actor 不用于初始化新 Actor；其 Refiner/空间 gate/delta/global-score
参数严格加载。拒绝缺失参数、PPO 输出冒充模板、或明显的 identity 初始化模板。

状态包含当前概率图、Base 图、熵、时间、固定分层/额外证据，以及原始 proposal logits `Z`
和累计空间融合门 `C`；全局输入包含当前/初始 global logits、全局证据和剩余预算。
当前配置是 29 个空间通道，128×128。原 Refiner 仍使用兼容的 26 通道状态；
额外 Z/C 通道只给新 Actor/Critic。它们是固定图像上下文的特征表示，不声称压缩特征完全充分。

动作是 STOP，或 `(16 个候选 ROI, 4 个证据层)`，共 65 个语义动作：

\[
\pi(\mathrm{STOP}|s)=p_o(0|s),\quad
\pi((i,l)|s)=p_o(1|s)p_r(i|s)p_l(l|s).
\]

ROI 根据每一步当前异常图和熵重新生成。只复用固定证据，不重新调用 VLM。
旧 gain_mean/lower/acceptance 头不决定执行，不作为 V；保留 Refiner 的空间软融合门。
REFINE 转移保持旧已训练 Refiner 的形式：

\[
Z'=Z+\Delta Z_{i,l},\quad C'=1-(1-C)(1-\alpha_{i,l}),
\quad P'_{HR}=(1-U(C'))P^0_{HR}+U(C')U(\mathrm{softmax}(Z')).
\]

注意先 softmax 再上采样。STOP 后 Z/C/P/global logits 均严格不变，进入终止状态；
达到 H=3 也是真终止，价值不 bootstrap。训练和评估的转移相同，没有训练软收益门/测试硬收益门差别。

使用原始输出分辨率（默认 512×512）上的质量函数：

\[
\Phi(P,g;Y,y)=-0.5\,\mathrm{CE}(g,y)
+y\,\mathrm{SoftDice}(P,Y)+(1-y)(1-\mathrm{mean}(P))
-0.2\frac{\sum(1-Y)P}{\max(\epsilon,\sum(1-Y))}.
\]

\[
r_t=\begin{cases}0,&\mathrm{STOP}\\
\Phi_{t+1}-\Phi_t-0.003,&\mathrm{REFINE}\end{cases},\quad
\gamma=1,\quad G_t=\sum_{k=t}^{T-1}r_k.
\]

因此 `G_0 = Phi_final - Phi_Base - .003 * N_refine`。GT 只在完整轨迹执行后计算奖励，
不进入 Actor/Critic/候选区域/状态。无二值化概率图、强制 Base 轨迹、首步强制 REFINE、
即时收益 STOP 辅助标签或 Refiner 监督损失。

Critic：独立 CNN → 全局平均/最大池化 → 拼接全局状态 → MLP → 单个无界线性实数。
输出层零初始化，避免初期任意有符号基线；不把它当执行阈值。

\[
V^\pi(s)=\mathbb E_\pi[G_t|s_t=s],\quad
L_V=\frac12\mathbb E[(V_\nu(s_t)-\operatorname{sg}(G_t))^2].
\]

默认 GAE lambda=1，即完整回报 `A=G-V_old`。代码支持完整终止回合上的 GAE，
但一体化脚本固定 gamma=lambda=1，本轮不额外扫参。

\[
\rho_t=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t)),
\quad L_\pi=-\mathbb E[\min(\rho_t A_t,
\mathrm{clip}(\rho_t,.8,1.2)A_t)].
\]

\[
H(\pi)=H(p_o)+p_o(1)[H(p_r)+H(p_l)],\quad
L=L_\pi+0.5L_V-\beta H(\pi).
\]

不对优势做组内中心化，不用当前更新后的 V 重算优势。beta 从 .01 按 epoch 线性降到 0。
Actor/critic 独立 Adam，学习率 1e-4/3e-4，梯度范数限制 1。
每张图 4 条自然采样完整轨迹，每批做 4 轮 PPO，小批量 16 个有效决策；随后丢弃该批。
“同预算”指相同图数、轨迹数、最大步长、PPO 轮数；实际有效决策数和 optimizer steps 随 STOP 改变，均写入日志。
所有采样状态、候选 ROI 坐标顺序、动作、old logprob、old V、G/A 均保存并 detach；
PPO 更新**不调用 Refiner 重放，也不重新运行 ROI top-k**，避免相同值的 top-k 重排改变动作含义。
不使用长期 replay buffer，不通过价值损失更新 Refiner。

算法依据：[PPO 原论文](https://arxiv.org/abs/1707.06347)、[GAE 原论文](https://arxiv.org/abs/1506.02438)。
这里的图像状态、Refiner、质量函数与成本是本项目的具体设计，不是论文对涨点的保证。

## 双卡 4090 一体化运行

依赖沿用之前 G/R 成功运行的环境，不自动升级 torch。Bash、flock、现有数据/CLIP/DINO 权重需可用。
默认 GPU0 跑 V_seed0，GPU1 跑 MC_seed0；不是 DDP。多个 seed 顺次分配，每张卡保持一个训练进程。
使用 FP32，关闭 TF32/AMP，以便机制对照。默认 8 图×4 轨迹；PPO 每次反传 16 个有效状态。
空间候选掩码已向量化，减少每步 B×K 次 GPU→CPU 同步。
CPU/GPU LRU 上限每进程 4/12 GiB，GPU 实际上限还保留至少 8 GiB 的启动空闲余量。
缓存会随数据集/图像数量不同而未用满，不刻意占满 24GB；日志记录实际峰值。
本地完成了小规模真实 CUDA 测试，尚未在两张 4090 上做完整数据训练或确认峰值/耗时。

推荐复用 R 的完整 source/target Base 缓存（不会复用 R 的策略或改变旧目录）：

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main
nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  MARA_CKPT=./checkpoint/mara_visa_hfa3_20260908_225850/ckpt/mara_epoch_29.pth \
  SOURCE_WORK_DIR=./checkpoint/mara_reward_r_v1 \
  WORK_DIR=./checkpoint/mara_ppo_v1 \
  GPU_IDS=0,1 TRAIN_SEEDS=0 EPOCHS=30 \
  TRAIN_BATCH_SIZE=8 TRAJECTORIES=4 PPO_EPOCHS=4 MINIBATCH_SIZE=16 \
  GPU_CACHE_GB=12 CPU_CACHE_GB=4 CPU_THREADS=4 \
  bash run_exp_mara_ppo.sh > nohup_mara_ppo.out 2>&1 &
```

`MARA_CKPT` 必须与 `SOURCE_WORK_DIR` 导出 Base 使用的 checkpoint 一致；
若此前 R 工作目录名称不同，改成实际目录。没有旧缓存时删掉 `SOURCE_WORK_DIR` 行，自动双卡导出。
默认训练划分 seed=20261018、80/20，目标每类别 8 图，沿用 R，统一 GT 审计排除空异常标注。
source 是 VisA 带标签 TEST 池划分，只是 MARA 训练/留出，**不是完整系统独立验证，也不是无监督训练**。
Base 和冻结 Refiner 可能已经见过 source 留出样本。目标 GT 只用于最终评估，不用于更新/选 checkpoint。
要做完整目标评估，首次运行前设 `SAMPLES_PER_CATEGORY=0`，使用新目录并承担额外耗时。

启动后日志位置：

```bash
tail -f nohup_mara_ppo.out
tail -f checkpoint/mara_ppo_v1/train_shard_0.log
tail -f checkpoint/mara_ppo_v1/train_shard_1.log
```

任一 worker 失败后不进入后续阶段。每完整 epoch 原子保存 Actor/Critic、两个 optimizer、scheduler、
Python/NumPy/Torch/CUDA RNG。原命令重跑从上次完整 epoch 恢复；已完成任务不再训练。
不跳过样本、不静默减 batch、不自动换 checkpoint。中途更改种子/预算/科学配置必须用新 WORK_DIR。
若 OOM，先原命令加 `EXEC_GPU_CACHE_GB=0` 释放常驻 GPU 证据缓存（仅执行设置，不改实验）；
若仍 OOM，使用新目录调低 TRAIN_BATCH_SIZE 或 MINIBATCH_SIZE。

## 输出、检查与解释

只上传 `checkpoint/mara_ppo_v1/ppo_report.zip`。原始权重和大缓存不在报告里。

| 文件 | 内容 |
| --- | --- |
| `macro_metrics.csv` / `metrics.csv` | Base、V、MC 每步类别宏平均/分类别 PRO、P-AUROC、F1、I-AUROC、背景 FPR |
| `comparisons.csv` | 同 seed、相同样本的 V−MC；每组与 Base 差异在 metrics 中 |
| `rollout_summary.csv` | 正常/异常分组的首步 STOP、多步执行数、累计回报、负收益执行数 |
| `step_trace.csv` | 逐步动作、ROI、done、实际变化、HR 质量分项/奖励、V/G/A；全为最终 greedy 策略 |
| `training.csv` | epoch 训练回报、V loss、旧 V explained variance、策略 KL、clip fraction、梯度范数 |
| `value_probe.csv` | source 留出固定初始状态，32 条独立随机轨迹回报均值/标准误与 V 对比 |
| `checkpoints.json` | 训练前后/冻结 Refiner 参数哈希、运行时间、峰值显存 |

主评估沿用分层 greedy：先选 STOP/REFINE，再分别 argmax ROI/layer；它不等于 65 动作联合 argmax。
V 学的是采样策略下的期望，**不能用某一条 greedy 轨迹的 G 直接判定 V 拟合是否正确**。
看 `value_probe.csv` 的多轨迹 MC 对照；MC 组该列 V 恒为零是设计如此。
I-AUROC 沿用现有概率图最大值评分；奖励分类项使用 global logits，二者不同。
PRO 沿用已有 1000 阈值、最大 FPR=.3 定义；Dice 奖励改善不能直接替代 PRO 改善。

最终权重：`jobs/V_seed0/final.pth` 与 `jobs/MC_seed0/final.pth`。
含嵌入 Base 与完整 PPO 参数，版本标记 `mara_algorithm=mara_ppo_v1`；
`test_mara.py` 已支持严格分流加载。旧 G/R 脚本只能加载旧模板，不能对 PPO 输出做旧收益门诊断。
部署仅需 Base+Actor+Refiner，不需要 GT/奖励；当前 infer 为方便共用轨迹代码也计算 V，V 不控制执行。

验收顺序：先查冻结哈希/STOP 恒等/GT 审计，再查学习回报与 V 拟合，最后查 PRO/P-AUROC/正常误报。
即使 V loss 降低、多步增加、代理回报上升，也不能单独宣称异常分割涨点。
