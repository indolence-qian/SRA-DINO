# PPO-LocSafe：候选动作诊断 O 与定位／正常图风险优化

本轮实现 **O + P0/P1/P2/P3**，不覆盖 G、R 或旧 PPO 入口。不加入 VLM、LoRA，
不重训 Base/Refiner。若 O 显示候选动作能力不足，下一阶段才做 Refiner F0/F1；
本脚本不会悄悄切换结构或把失败组替换成 Base。

## 科学对照

| 组 | 相对上一组的改动 |
| --- | --- |
| P0 | 当前 PPO+V 原奖励，重训基线 |
| P1 | 只将奖励中的分类系数从 .5 置 0 |
| P2 | P1 加区域均衡、困难背景排序项，权重 .5 |
| P3 | P2 加终止时正常图风险成本，拉格朗日 PPO |

每个 seed 四组相同 Actor/Critic/Refiner 初始化哈希；Base/Refiner 冻结。
输入、ROI、空间融合门、无收益否决门、H=3、STOP=0、REFINE=.003 都相同。
默认 gamma=lambda=1，4 条自然轨迹/图，PPO 4 passes，batch=8，minibatch=16，
Actor/Critic LR=1e-4/3e-4，30 epoch，FP32，关闭 TF32/AMP。
Actor/Critic 仍不接收 GT；GT 只在 GT-free 完整轨迹执行之后构建奖励。

原质量是 `-.5*CE + q_loc -.2*q_bg`。P1 去掉 CE；q_loc 对异常为 Soft Dice，
正常为 `1-mean(P)`。P2/P3 对异常再减去 `.5*L_rank`：

\[
L_{rank}=\frac1K\sum_{k=1}^K E_{u\in C_k,v\in B_h}[.1+P(v)-P(u)]_+.
\]

GT 在原始输出分辨率（默认512）上分 8 连通区域，区域等权，不按缺陷面积加权。
B_h 是 GT 背景中异常概率最高的5%；前景每区域最多128个等距点，困难背景排序值
最多128个等距点，确定性近似，不消耗行为/PPO随机数。不丢弃小连通区域。
全前景 GT 没有背景对，rank=0；空异常 GT 由统一审计排除。
**rank 不是 PRO，也没有保证提高 PRO。**

设 `B(P)=MeanTop5%(P)`，P3 终止风险为：

\[
C_T=1_{y=0}[B(P_T)-B(P_0)]_+,\quad E[C_T]\le\rho_N\epsilon,\quad\epsilon=.001.
\]

rho_N 是本轮实际源域 fit 的正常图比例。因此这是正常图条件平均风险预算 .001，
不是把全体图像平均错误地当正常图平均。每步奖励为定位质量差减 REFINE 成本；
仅在该图真正终止（主动 STOP 或 H 上限）那一步再减 `lambda*C_T`，只结算一次。
允许前面步骤暂时负奖励，不进行即时收益门否决。

lambda 初值1，每完整 epoch 训练后计算
`lambda_next=max(0,lambda+10*rho_N*(mean_normal_cost-.001))`，
下一 epoch 才启用。保存的当前 epoch 权重、V 校准和源域验证均使用训练该 epoch 的
lambda；`dual_used/dual_next` 都记录。lambda/optimizer/RNG 全部支持精确 epoch 恢复。
没有额外网络，仍是 Actor+V，V 拟合本组完整 MC G；不是 CPO 算法或安全保证。

## 数据边界与选模

先用与历史 R/PPO 相同的源域 split seed=20261018、80/20，得到历史训练池和历史留出。
再在历史训练池里按类别/真实标签、seed=20261020 分约80/20 fit/DEV。
原数据规模约为1384 fit、346 DEV、432历史源域留出，实际数量以审计/round规则为准。
三者严格不重叠；历史留出不更新参数、不选 checkpoint。
这只是 MARA 层面划分：冻结 Base/Refiner 可能见过这些源域图，**不是全系统独立验证或无监督训练**。

DEV 在 epoch 0、1、5、10、15、20、25、30 做全量验证（不是96张小面板）。
候选 epoch 排除0；要求 PRO>同样本Base、背景FPR@.5增量<=.0002（.02个百分点）、
P-AUROC下降<=.002（.2个百分点）。在合格者中选 PRO 最大、并列时较早 epoch。
若全不合格，仍评估最终权重供诊断，但明确标记 `FAILED_guardrails_FINAL_diagnostic_only`；
不自动换 Base，也不宣称成功。保存 `final.pth`（最后epoch）与 `selected.pth`（源域规则选择）。
主结果评估 selected，selection.json 列出所有候选源域指标，便于检查选模过程。

**所有训练组完成源域选模后，脚本才进入任何目标域评估。**
目标不选系数/epoch/阈值。默认目标全量 `SAMPLES_PER_CATEGORY=0`。
同一前向输出分别统计 source_dev、source_historical、target_full、target_historical、target_additional，
不把角色混成一个源域指标。历史目标 ID 取缓存来源 manifest 的交集；
target_additional 仅表示“不在该 manifest”，不能保证用户从未查看或类别从未见过。
若不用缓存来源，历史目标追踪不可用，不能据此声称目标完全独立。

## O：先确认现有动作是否有用

源域 DEV 每类别固定最多8图（正常/异常交替），枚举 STOP+16ROI*4层。
在每类别最多2图上进一步做宽度8、最长H步的 beam search；ROI随每一步当前状态重生成。
按 `Phi_P2(final)-Phi_P2(Base)-.003*N_refine` 排序，STOP也保留。
保存全部候选的动作、代理收益、rank、概率变化，以及最佳一/多步热图；不保存大张量。
如果来源是旧 PPO 目录且 V_seed0/final.pth 存在，自动加入旧策略热图对照。
旧 Actor 可能在这些 DEV 图上训练过，此对照不是独立泛化成绩。

**这是 GT-assisted 有限代理目标搜索，不是部署方法，也不是 PRO 全局上界。**
候选热图统一按真实 PRO 评估；beam没找到不等于穷举证明不存在“先亏后赚”。
O 输出建议证据，不用小面板代理分数自动终止/改变四组预算。可设置 ORACLE_ONLY=1 先只跑O。

## 双4090一体化启动

在已有成功 G/R/PPO 的 `qfg_addino` 环境中运行，不自动升级torch/安装依赖/下载权重。
SOURCE_WORK_DIR 是服务器上的完整旧目录，不是下载的小报告zip；只读复用缓存，缺的图才前向Base。
默认每卡一个独立任务，不是DDP：GPU0顺次 P0/P2，GPU1顺次 P1/P3。
多seed固定分配两个worker；GPU缓存每进程上限12GiB，还动态保留至少8GiB启动空闲显存。

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main
nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  MARA_CKPT=./checkpoint/mara_visa_hfa3_20260908_225850/ckpt/mara_epoch_29.pth \
  SOURCE_WORK_DIR=./checkpoint/mara_ppo_v1 \
  WORK_DIR=./checkpoint/mara_ppo_locsafe_v1 \
  GPU_IDS=0,1 TRAIN_SEEDS=0 EPOCHS=30 SAMPLES_PER_CATEGORY=0 \
  TRAIN_BATCH_SIZE=8 TRAJECTORIES=4 PPO_EPOCHS=4 MINIBATCH_SIZE=16 \
  GPU_CACHE_GB=12 CPU_CACHE_GB=4 CPU_THREADS=4 \
  bash run_exp_mara_locsafe.sh > nohup_mara_locsafe.out 2>&1 &
```

若实际旧 PPO 目录不同，修改 SOURCE_WORK_DIR；没有旧目录就删掉该行。
原始 MARA_CKPT 不可换成旧 PPO final.pth，后者只可作为 REFERENCE_PPO_CKPT。
仅先诊断可在原命令加 ORACLE_ONLY=1；看完 oracle_report.zip 后原命令改0会复用O并继续。
最快机制试跑可首次用新目录设 SAMPLES_PER_CATEGORY=8；那只是历史子集，不是完整泛化验证。
确认实验首次另用新目录设 `TRAIN_SEEDS="0 1 2"`，完整报告均值/标准差，不选最优seed。

OOM 不偷偷减 batch 或跳图。先原命令增加 EXEC_GPU_CACHE_GB=0 释放缓存后重跑；
改变训练batch/样本/奖励/seed/预算必须新目录。普通中断同提交同命令重跑自动恢复。
若仅日志代码更新后续跑，可明确用 RESUME_PREPARED=1：保留原封存配置，忽略所有新科学参数。
一个worker失败会停止同阶段另一个worker，不进入下阶段；失败日志打印尾部。
本地小规模CUDA/合成流程不能代替服务器双4090完整显存/耗时或性能验证，不承诺涨点。

## 中间过程与结果位置

```bash
tail -f nohup_mara_locsafe.out
tail -f checkpoint/mara_ppo_locsafe_v1/oracle_shard_0.log
tail -f checkpoint/mara_ppo_locsafe_v1/train_shard_0.log
tail -f checkpoint/mara_ppo_locsafe_v1/train_shard_1.log
```

训练时打开 `jobs/P0_seed0/monitor/index.html`（其他组对应各自目录）。
每epoch刷新原始loss、batch/update、KL、熵、梯度、层/ROI选择、正常/异常收益、全epoch尾部风险、
lambda与统一P0参考回报。验证epoch固定原图/GT/Base/每步P/deltaP/空间门/ROI路径，不挑好图。
完整训练结束后 `visual_report/index.html` 有组间总览和独立MC V校准（每状态32条随机轨迹）。
V不等于单条贪心轨迹回报；不同组原始奖励定义不同，不按reward/loss大小宣称优劣。

最终只上传 `checkpoint/mara_ppo_locsafe_v1/locsafe_report.zip`；
若先只诊断，则上传 `oracle_report.zip`。无证据缓存、权重或全量概率NPZ。
主表含role、category、seed、每步指标；step_trace含统一参考回报、rank、tail、终止风险和V/G/A。
risk_summary按正常/异常、源/目标角色分组；value_calibration_summary提供正常/异常各自的
MAE、RMSE、R2及明显乐观误判计数，散点图也按标签区分。选模失败的标志不能忽略。
排序奖励增加不保证 PRO 增加；全部STOP、与Base持平也不算有效改进。
