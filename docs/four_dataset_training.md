# 训练当前网络并统一测试四个数据集

入口：`run_train_test_four_datasets.sh` / `train_test_four_datasets.py`。
本入口对应 **DINO 正常参考残差＋约束器/Flow＋高分辨率＋文本**，
不是 `run_test_ld.sh` 对应的早期 CLIP+DINO / MARA 网络。
无需提供已训练的 checkpoint，只需要原来的 DINO/CLIP 预训练权重和数据。

## 网络与数据协议

- 保持冻结的原始 DINOv3-L/16，提取第 5/11/17/23 层；不添加 HFA/Adapter/MARA。
- 每张查询与 K=4 张同类别正常 TRAIN 图，在完整 1024 维空间先匹配、相减，再投影256维。
- 使用一张全局图和四张原生图局部裁剪；全局/局部晚融合权重0.75/0.25。
- 从头训练一套原有约束器＋8层 affine coupling Flow，以及原有逐层无偏置文本 MLP。
- 文本默认采用最近对比实验中的 T2 普通正常/异常 logit 差，融合 λ=0.5。
  `TEXT_READOUT=T1` 可切换为原始单对单位差分；两种都是已有读出，不新增网络模块。
  本轮只训练指定的一种，不重跑 T1/T2/T3 或 A/B/C/D 消融。
- 默认训练15轮；Flow和文本分别按 VisA 留出类别上的宏平均 `(AUPRO + P-AP)/2` 选择已训练轮次。
  固定阈值来自源正常 CAL 图；CAL图不参与梯度训练，不按四个目标测试集选轮次或调融合。
- 训练源为 VisA TEST 中的正常/异常及其标签，按源类别划分 train/val；VisA TRAIN正常图作参考。
  这是源端有监督、目标允许正常参考的跨数据集评价，不是源端无监督或无参考零样本。
- 测试集固定为 MVTec、BTAD、MPDD、LD。每个目标只从正常 TRAIN 取同类别 K 张参考，
  目标异常图不参与梯度更新。LD 的原 val=251 张作为 test，原 train 只提供标注正常参考。
- MVTec/BTAD/MPDD 已用于既有方法开发，这次是统一重训后对照；LD 是新加入的目标，
  不把此前三个基准重新宣称为未参与设计的盲测集。

网络、损失、源选模、训练断点和推理计算复用当前残差 D/HR 代码。
新目录使用单独协议和哈希；原封存代码、训练结果和运行入口不修改。

## 服务器一条命令开始

先保证上传的 LD 压缩包或解压数据位于 `/mnt/qfg/Tate_qfg/data/`，
原有 VisA/MVTec/BTAD/MPDD、DINO 与 CLIP 权重仍在原目录。

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main
conda activate qfg_addino
mkdir -p logs

nohup env GPU_IDS=0,1 EPOCHS=15 \
  WORK_DIR=./checkpoint/residual_hr_text_four_datasets_v1 \
  bash run_train_test_four_datasets.sh \
  > logs/residual_hr_text_four_datasets_v1.log 2>&1 &
```

不需要设置 `WEIGHT_PATH`、`BASE_CKPT` 或 `MARA_CKPT`。
脚本自动从当前预训练 DINO/CLIP 开始训练新头，不复用旧已训练的 Flow 或文本头。
两张 GPU 在特征导出/测试时按类别分片；训练时 GPU0 训练 Flow、GPU1 训练文本头。
只有一张卡时设 `GPU_IDS=0`，两个头会顺序训练。

查看进度：

```bash
tail -f logs/residual_hr_text_four_datasets_v1.log
```

如果之前运行过数据包安装器，拉取代码提示旧适配文件会被覆盖，可先按
[LD安装说明](ld_dataset.md#一体化测试现有网络)将那些旧适配文件单独存入 stash。

## 默认路径

| 内容 | 路径 |
|---|---|
| VisA训练源 | `/mnt/qfg/Tate_qfg/data/VisA` |
| MVTec测试 | `/mnt/qfg/Tate_qfg/data/mvtec` |
| BTAD测试 | `/mnt/qfg/Tate_qfg/data/BTAD/BTech_Dataset_transformed` |
| MPDD测试 | `/mnt/qfg/Tate_qfg/data/MPDD` |
| LD测试 | `/mnt/qfg/Tate_qfg/data/LD_AD_20261116` |
| DINO预训练 | `./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` |
| CLIP预训练 | `./CLIP/ckpt/ViT-L-14-336px.pt` |

LD目录未解压、但同层已有 `LD_AD_20261116_bundle.tar.gz` 时，脚本会只提取数据树；
存在 `.sha256` 时先核验整包，不用旧包适配器覆盖新代码。
其他数据根目录可用 `DATA_ROOTS_JSON` 覆盖，例如：

```bash
DATA_ROOTS_JSON='{"visa":"/your/VisA","mvtec":"/your/mvtec","btad":"/your/BTAD","mpdd":"/your/MPDD","ld":"/your/LD_AD_20261116"}'
```

四个测试集默认全部图片都参与，不取固定数量子集。正常参考排除查询的 RGB 重复，
不同类别的参考不能共享相同内容。新资产不允许悄悄缺失某个测试集或加入源训练。

## 运行过程与断点续跑

1. 检查原训练环境、GPU数量、预训练权重和四个目标目录；建立完整清单与正常参考。
2. 导出 VisA 源特征，只保存残差及对应关系，不保存额外的原图外观缓存。
3. 固定源正常 RMS，编码冻结 CLIP 文本，训练新 Flow/文本头并做源类别选模。
4. 全部源训练与选择结束后，导出四个测试集特征，统一使用同一选定模型推理。
5. 所有测试预测完成并锁定哈希后，才解码目标 GT，生成分数据集、逐类别指标和可视化。
6. 输出一个汇总表与小型结果 ZIP。

同一目录、同一配置重新执行相同命令会复用已完成特征，恢复模型/优化器/学习率状态，
复用已锁定预测。中断后先确认原进程已经退出；目录锁会防止两次运行同时写入。
更改网络配置、数据、源划分或代码需要新 `WORK_DIR`，不覆盖旧模型和封存缓存。
若固定计划15轮，首次 `TRAIN_UNTIL_EPOCH=5` 后可沿用同一目录继续到15轮；
中间快照和每个 horizon 的评价单独保存。默认直接跑满15轮。

可调参数沿用现有设置：

| 环境变量 | 默认 | 用途 |
|---|---|---|
| GPU_IDS | 0,1 | 可用GPU，不能重复 |
| EPOCHS / TRAIN_UNTIL_EPOCH | 15 / EPOCHS | 训练计划与本次停止轮次 |
| TRAIN_SEED / DATA_SEED | 42 / 42 | 模型初始化及固定划分 |
| TEXT_READOUT | T2 | T2普通正常/异常；T1原始单对差分 |
| REFERENCE_COUNT | 4 | 正常参考数量，可1/2/4 |
| BATCH_SIZE / MICRO_BATCH_SIZE | 8 / 1 | 残差头批量和显存微批量 |
| LOADER_WORKERS | 2 | 训练加载worker数 |
| IMAGE_SIZE / EVAL_SIZE | 512 / 512 | DINO输入和最终评价分辨率 |
| SELECTION_SIZE | 256 | 源选模评价分辨率 |
| TEXT_STRENGTH / HIGH_WEIGHT | 0.5 / 0.25 | 固定文本增量及局部分支权重 |
| DINO_WEIGHTS / CLIP_WEIGHTS | 上述本地文件 | 原始预训练权重 |
| RUN_EXPORT / RUN_TRAIN / RUN_PREDICT / RUN_EVAL | 1 | 断点阶段控制，缺少必须的封存文件会报错 |

缓存只保留残差；prepare会打印存储估算，还需为模型和预测预留空间。
已有单轮消融大缓存不会被覆盖或自动删除。本入口也不更改原文档里的旧实测结果。

## 直接看四个数据集效果

本轮结果在：

```text
checkpoint/residual_hr_text_four_datasets_v1/
  reports/FOUR_DATASETS.md       四数据集主表
  reports/summary.csv            数据集类别宏平均、有效类别数
  reports/per_category.csv       逐类别结果
  reports/summary.json
  heads/seed42/D/density/        新约束器/Flow权重及训练历史
  heads/seed42/D/text/           新文本头权重及训练历史
  decisions/                    源选中轮次、校准和固定阈值
  metrics/seed42/through15/      每类别的完整评价与mask审计
  panels/                       固定抽样热图
  four_datasets_report.zip       可下载的小型结果包
```

若想生成全部测试图片的彩色热力图、独立热图与原图叠加图，使用
[离线热力图导出脚本](four_dataset_heatmaps.md)。它只读取已有预测，在CPU运行，
不需要重新训练，也不更改本实验的封存代码和指标。

主表 `full` 是同一完整模型在四个数据集上的效果；`visual` 是同模型 Flow 分支，
用于看文本带来的变化，并不表示另外训练了一套网络。
报告 I-AUROC、I-AP、P-AUROC、P-AP、AUPRO、F1-best，以及源阈值下的背景FPR和像素召回。
不把类别数不同的数据集直接混成一个平均分。

评价沿用最新修正版：图像标签取清单，与缩放后的mask前景独立；
原始mask按非零值解码，缩小时保留微小前景，BTAD官方全零mask原样保留，不丢弃异常图。
LD的标注是矩形框代理mask，像素指标必须注明框级口径，不与精确分割真值等同解释。
Cam5没有正常测试图，单类图像AUROC为 N/A，并从有效类别平均中排除。
F1-best仅作测试描述统计，实际固定阈值仍来自源CAL。

本地检查使用小型真实 Flow/文本头训练和测试专用编码器，验证四数据集完整链路、
目标GT隔离、BTAD零mask、单类AUROC、续跑和旧协议隔离。服务器实际训练出的数值不预先填造。
