# 正常残差池：先知道正常图之间会怎样不同

入口：`run_exp_normal_pool.sh`，独立于旧 `run_exp_residual_text.sh`。
这是下一轮的第一步：**真实正常残差池 + 正常变化尺度校正**。本轮不训练新的小头、
Flow或文本，不运行RL/VLM，不把统计估计称为训练后的残差学习，也不宣称已经得到语义证据。
原来的Flow/文本入口、权重、缓存均保留，后面再接入正常池校正后的向量。

## 实际做了什么

1. 同类K张正常图分别走整图512、四个原图裁剪（各输入512），冻结DINOv3提取多层特征。
2. 每张正常图轮流作为查询，只匹配**其他正常图**。其自身的整图及所有裁剪均排除。
   在完整1024维上匹配并相减，再固定投影为256维。保留符号，不单位长度归一化。
3. `normal_pools/<dataset>__<category>.npz`实际保存全部正常残差向量：
   `residual[K,5,L,D,h,w]`、原始距离、最近邻索引、逐层/全局局部/逐通道二阶矩。
   对应JSON记录每个正常查询使用了哪些参考、向量形状、文件哈希与统计量。
4. 从源TRAIN类别的池子，按类别等权得到源正常变化先验；不混入源VAL或目标类别。
5. 查询图仍匹配全部K张正常图，得到残差。以正常池的通道尺度校正它：正常经常变化的
   方向敏感度降低，正常稳定的方向保留敏感度。不是随便再减一个正常残差向量。
6. 各视图独立打分，局部按坐标Hann拼接，全局/局部按0.75/0.25融合。

校正使用源先验与当前类别正常池二阶矩的混合：
`v = rho * v_source + (1-rho) * v_support`，`r_corrected = r / sqrt(max(v, floor))`。
默认`rho=0.5`，不按目标指标寻找最优值。设置源先验相关下限，防止少量参考导致近零尺度放大噪声。
不减均值，因此零残差经过校正仍为零。不使用逐测试图min-max、测试集联合统计或目标梯度更新。

**K=4时正常互比使用3张、查询使用4张**。这不是同一个残差分布，也不假设它们严格一致。
本实验将LOO统计作为有收缩的变化先验，分数校准使用独立源正常查询对全部K参考的真实分数，
再在源VAL未见类别检验。不能把LOO的分位数直接当成“目标误报保证为1%”的阈值。
本版支持K=2/4；K=1不能真实做正常图互比，直接报错，不用同一张图伪造正常池。

## 八组对照与读结果

| 前缀 | 含义 | 对照目的 |
|---|---|---|
| D | 完整1024维原始距离 | 原距离基线 |
| P | 投影256维残差大小 | 分离压缩影响 |
| S | 仅源域逐通道正常尺度 | 逐通道尺度的作用 |
| N | 源域先验 + 当前类别正常残差池 | 正常支持池的额外作用 |

每组均输出`_G`整图、`_HR`整图加局部。所有组共享查询、正常参考和DINO特征。
没有自动回退，不把S改名为N，也不把更差组藏掉。显式设置`POOL_PRIOR_WEIGHT=1`时N与S相同是预期控制。

重点看`comparisons.csv`中的`support_pool_increment`（N_HR减S_HR）和
`total_pool_increment`（N_HR减P_HR）；同时看AP、PRO、实际背景误报和异常召回，不能只看AUROC。
`I_AUROC`固定使用最大值；`I_AUROC_topmean`固定使用最高1%像素均值，是单独诊断，
不根据目标指标挑选更高者当“正式结果”。两者使用相同分割图。

## 数据边界

- 默认VisA作为源数据，类别留出划分沿用V1（9类TRAIN、3类VAL），目标为MVTec/BTAD/MPDD。
- 只从源TRAIN的**完整正常查询图**选最多64张、类别均衡的CAL，不使用异常图背景mask校准。
  CAL不参加梯度训练；本次本身没有梯度训练。源VAL只作独立诊断，不参与先验/阈值拟合。
- 每个目标类别固定K张正常TRAIN参考。正常与查询RGB内容去重，目标异常不进池子。
- 目标预测全部锁定后，评估才打开目标mask。mask也不参与池参数、rho或top百分比选择。
- 新实验的CAL与旧V1不同，所以D组也需按新CAL重新校准。不能声称与旧D数值逐位一致，
  应在本轮八组内部比较，不用新校准结果冒充对旧方法完全同口径的改进。
- 单一`DATA_SEED`不代表稳健性。后续用新目录做多组参考；不把已参与开发的目标集当最终未见验证。

## 一体化运行（Linux服务器）

使用原来能运行DINO实验的环境，无新增依赖，不需要CLIP权重或Base检查点。

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main
conda activate qfg_addino

nohup env \
  TRAIN_PYTHON="$(command -v python)" GPU_IDS=0,1 \
  WORK_DIR=./checkpoint/normal_residual_pool_v1 \
  REUSE_DIR=./checkpoint/residual_first_hr_text_v1 \
  DINO_REPO_DIR=./dinov3 \
  DINO_WEIGHTS=./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  REFERENCE_COUNT=4 DATA_SEED=42 \
  bash run_exp_normal_pool.sh > normal_pool_nohup.out 2>&1 &
```

`REUSE_DIR`是原实验在服务器上的**完整工作目录**，不是下载的analysis.zip。里面必须还在
`features/`、`sealed_source.json`、`sealed_eval.json`和原配置；新流程只读这些文件，不复制几十GB查询缓存。
仍需重新编码正常参考并构建正常互比残差池；这部分上一轮没有。
会验证原配置代码哈希、骨干/投影/精度/裁剪/K/数据种子、参考ID及缓存封存哈希，不做宽松兼容。
旧模型权重/选模结果不复用。旧缓存缺失或设置不匹配时明确停止，不静默用错参考。

如果没有原特征缓存，改成`REUSE_DIR=none`，从原始数据重新提取。默认`REUSE_DIR=auto`
仅在默认V1目录存在时尝试只读复用，否则从头提取。非默认数据位置用`DATA_ROOTS_JSON`
（沿用旧入口），例如`'{"visa":"/data/VisA","mvtec":"/data/mvtec","btad":"/data/BTAD","mpdd":"/data/MPDD"}'`。
单卡用`GPU_IDS=0`。更改GPU数量、参考集、rho或几何设置需新WORK_DIR，不能混用缓存。

完整运行顺序：准备 → 源正常池/残差 → 源CAL拟合 → 源VAL预测封存/评价 →
目标正常池/残差 → 全部目标预测封存 → 目标评价 → 汇总 → 自动分析包。
任一worker失败则后续阶段不启动；同一WORK_DIR用flock防止并发覆盖。

中断后使用完全相同命令恢复；完整导出完成后如仅重跑评价，可加
`RUN_EXPORT=0 RUN_PREDICT=0`。新建目录时不要关闭必需阶段。
首次prepare会估算磁盘空间；复用旧缓存仍需为八组预测图和新正常池预留约数十GB空间。

## 结果位置

```text
WORK_DIR/
  normal_pools/             # 真正的正常残差向量池 + 自排除对应关系 + 统计量
  normal_model.json         # 仅源TRAIN正常先验/CAL缩放/阈值，不是神经网络权重
  reports/source/README.md  # 类别留出的源VAL结果
  reports/eval/README.md    # 三套目标数据的八组汇总
  reports/eval/metrics.csv  # 逐类别全部指标
  reports/eval/comparisons.csv
  panels/eval/              # 固定哈希抽样图，不按好坏挑图；颜色不是二值阈值
  normal_pool_analysis_*.zip
```

分析包包括配置、正常池统计/对应关系、预测锁、指标、对照图、日志，不包括巨大的原始向量池、
查询残差缓存或预测张量。它能支持结果分析，但不能单靠这个小包独立复算全部像素指标。

本地检查（CPU仅使用测试专用特征，不是真实模型效果）：

```powershell
$env:SRA_TEST_BASH='D:/qfg/编译器/Git/bin/bash.exe'
& 'D:/qfg/python/Scripts/python.exe' -m unittest discover -s tests -p test_normal_pool.py -v
```
