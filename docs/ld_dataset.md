# LD 数据集转换与服务器安装

原始目录：`D:\project\codex\LD_数据集资产_20261116`。
转换输出：项目内 `data/LD_AD_20261116/`。原始文件不修改。
服务器默认数据根目录：`/mnt/qfg/Tate_qfg/data/LD_AD_20261116`。
数据集参数：`--dataset ld`，可通过环境变量 `LD_DATA_ROOT` 覆盖根目录。

**若要重新训练当前的 DINO 残差＋Flow＋高分辨率＋文本网络，并统一测试
MVTec/BTAD/MPDD/LD，请使用 [四数据集训练测试入口](four_dataset_training.md)。**
下方 `run_test_ld.sh` 是保留的 CLIP+DINO / MARA 已有权重测试入口，不用于最新残差网络。

## 实际数据与使用口径

以实际图片、YOLO 标签和 `split.csv` 三者一致性为准，不按文件名的 `OK/NG` 判断。
原始 train 全部保留为 train，原始 val 全部保留为 test，未重新随机划分。
这批 test 是原资产的验证集，不是新增的独立盲测集。数据集不提供额外 VAL，
加载 `DatasetSplit.VAL` 会明确报错，避免把同一批数据同时当验证和测试。

| 原任务 | 转换类别 | train 正常/缺陷 | test 正常/缺陷 | 合计 |
|---|---|---:|---:|---:|
| 齿面 | gear_teeth | 136 / 134 | 50 / 50 | 370 |
| 轴面26 | shaft_cam26 | 281 / 112 | 17 / 16 | 426 |
| 轴面5only | shaft_cam5 | 128 / 663 | 0 / 107 | 898 |
| 轴面9only | shaft_cam9 | 17 / 22 | 2 / 9 | 50 |
| 合计 | 4 类 | 562 / 931 | 69 / 182 | 1744 |

“正常”指该图没有标注真实缺陷框，不代表经独立复检确认的好品。
说明文档中提到的独立好品误报测试集、实拍 NG 评测集不在本次输入目录中。
Cam9 的原说明 train=36 已过时，实际文件及 CSV 一致为 train=39 / val=11。
1,744 张图全部保留，且 896 张图的文件名 OK/NG 与上述标签定义不一致；
例如空标签但文件名 NG，以及有缺陷框但文件名 OK，均保留并记录，不按名称重新打标签。

原类别 0/1/2/3/4/6 都合并为二值真实缺陷。根据原说明，
类别 5 `bscyichang` 为应忽略的反光等干扰，不画入异常掩码；
只有类别 5 的图片计入标注正常。若同图同时有真实缺陷和类别 5，
图像为缺陷图，掩码仅覆盖真实缺陷框；真实缺陷框覆盖的重叠区域仍为前景。
原七类 ID、归一化框坐标、像素框坐标和完整原始 TXT 均随包保留。

## 标注和评价限制

源数据只有 YOLO 矩形框，没有精确分割标签。
转换得到 **1,113 张框级代理掩码**，每张掩码与对应原图同尺寸、L 模式、值为 0/255。
框的左右边界分别向下/向上取整，以半开区间填充，异常掩码取真实缺陷框的并集。
不缩放、不裁剪、不重新编码 JPG，适配器读取时才使用原有网络的 Resize/CenterCrop。
掩码读取使用最近邻，图像级标签独立于缩放后掩码是否仍有前景。

像素 AUROC、PRO、像素 F1 只能称为“框级代理标注上的定位指标”，不能当作精确分割性能。
Cam5 test 全部是缺陷图，其单类别图像级 AUROC **未定义**；旧测试入口可能打印 0，
这不表示真实性能为 0，不能直接把该 0 纳入四类别 I-AUROC 平均来解释性能。
Cam5 可计算框级定位指标；整体图像指标可另算全测试集，或对有两类图像的三类别报告宏平均，
但必须说明采用哪一种口径。未改变既有测试脚本的指标实现；新增一体化入口单独修正此报告口径。
原图尺寸混有 2048x1080、4096x2160、2800x2400 和 2800x2528，
不能按说明中的固定尺寸反推框位置。转换始终读取每张图的实际尺寸。

## 输出结构

```text
LD_AD_20261116/
  gear_teeth/                     # 其余三个类别结构相同
    train/good/*.jpg
    train/defect/*.jpg
    test/good/*.jpg
    test/defect/*.jpg
    ground_truth/defect/*_mask.png
    ground_truth/train/defect/*_mask.png
  shaft_cam26/
  shaft_cam5/
  shaft_cam9/
  annotations/
    original_dataset_notes.md
    gear_teeth/original_split.csv
    gear_teeth/train/*.txt
    gear_teeth/val/*.txt
    ...
  manifest.json                   # 相对路径、标签、组、框和每个文件的 SHA256
  audit_report.json               # 实际计数、分辨率、问题与限制
  annotation_preview.jpg         # 红色真实缺陷框；青色为干扰框
```

标准接口保持 `image/mask/is_anomaly/image_path` 以及
`data_to_iterate = [category, state, image_path, mask_path]`。
TRAIN 的缺陷样本也提供真实的框级掩码，正常参考可以通过 state=good 单独筛选。
`test.py`、`test2.py`、`test_mara.py` 通过注册表读取 `ld`。
只新增数据集接口，不改变网络或已有实验的封存代码与权重。
其他带固定数据集白名单的实验入口未在本次接入范围内。

## 一体化测试现有网络

入口：`run_test_ld.sh` / `evaluate_ld.py`。保持当前 CLIP+DINO 网络，支持原 Base
以及包含该 Base 的 MARA / PPO MARA 权重。它复用 `train_up.py`、`train_mara.py`、
`test_mara.py` 的现有构建函数和 `tools/utils_up.py` 的前向计算；MARA 仍调用原 agent 的
`infer(sample=False)`，门控、步数和融合按权重元数据恢复，不重新设置门控阈值。
所有已加载模块冻结并切换到 eval，网络不接收测试 GT，也不计算 GT oracle。
单塔 DINO、Flow 头或文本消融头不属于这个入口，传入会明确报错，不会被改造成别的网络。
权重字段/尺寸必须与当前模块严格匹配，避免部分加载留下随机参数。

先在服务器拉取最新代码、激活原环境。**当前权重只在你的服务器上，
`WEIGHT_PATH` 必须指定实际文件或 ckpt 目录，不能照抄示例假路径。**
下面通过终端输入实际地址，不需要编辑脚本：

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main
conda activate qfg_addino
read -r -p '请输入现有 checkpoint 文件或 ckpt 目录的完整路径: ' WEIGHT_PATH

mkdir -p logs
nohup env WEIGHT_PATH="$WEIGHT_PATH" GPU_ID=0 \
  WORK_DIR=./results/ld_current_network_v1 \
  bash run_test_ld.sh > logs/ld_current_network_v1.log 2>&1 &
```

如果之前在服务器运行过压缩包里的安装器，`git pull` 可能提示本地适配文件被覆盖。
先把这些旧适配文件单独存入 stash，再拉取；数据和权重不进入这个命令的范围：

```bash
git stash push --all -m 'LD adapter before integrated runner' -- \
  Datasets/__init__.py Datasets/ld.py tools/check_ld_dataset.py tools/prepare_ld_dataset.py
git pull --ff-only origin main
```

保留该 stash 供查看；新版仓库已经包含 LD 适配器，不需要再执行旧包安装器或自动恢复旧补丁。
如曾自定义其他数据集路径，请从 stash 核对恢复所需路径。

数据默认读取 `/mnt/qfg/Tate_qfg/data/LD_AD_20261116`。
如果该目录尚未解压，但同层已有原 `.tar.gz`，脚本会只解压数据目录，
并在存在 `.sha256` 文件时核对整包校验和；不会用旧包里的适配代码覆盖仓库文件。
默认再全量核验原图、标签和框掩码。仅 test 的 251 张图片用于评价，train 不参与推理。
本入口不需要重新打包或重新上传已有数据。

权重是目录时，只在该目录选择**最大数字 epoch**，不读取测试成绩、不根据文件修改时间选模。
同时含 Base/MARA 命名的目录需要显式 `MODE=base` / `MODE=mara` 或精确文件路径。
默认只测试一个 checkpoint；MARA 同时报告其内置 Base 和 MARA，Base 权重则只报告 Base。
`MARA_CKPT` / `BASE_CKPT` 可作为 `WEIGHT_PATH` 的兼容别名。

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| WEIGHT_PATH | 必填 | 现有 checkpoint 文件或 ckpt 目录 |
| MODE | auto | 由 checkpoint 判定 base/mara，也可显式指定 |
| GPU_ID | 0 | 使用的一张物理 GPU；程序内映射为 cuda:0 |
| LD_DATA_ROOT | `/mnt/qfg/Tate_qfg/data/LD_AD_20261116` | 转换数据根目录 |
| WORK_DIR | `./results/ld_时间戳` | 输出目录；续跑需保持同一路径 |
| BATCH_SIZE / NUM_WORKERS | 4 / 2 | 加载批量和 worker 数，显存不足可降低批量 |
| IMAGE_SIZE | 512 | 现有推理尺寸；默认保持不变 |
| DINO_WEIGHTS | 项目根目录的 DINOv3 `.pth` | 原始本地预训练权重 |
| CLIP_WEIGHTS | `./CLIP/ckpt/ViT-L-14-336px.pt` | 原始本地 CLIP 权重；不自动下载 |
| HFA_SETTING | hfa3 | 只在旧权重缺少 HFA 元数据时使用的后备设置 |
| VERIFY_DATA | 1 | 全量文件核验；0 仅用于已核验数据的续跑 |
| CHECK_ONLY | 0 | 1 只核验数据、权重字段及路径，不运行网络 |
| SAVE_VIS / VIS_PER_CATEGORY | 1 / 8 | 保存可视化及每类数量 |
| THRESHOLD | 0.5 | 固定逐图报告阈值；不根据 LD 测试成绩调参 |
| PRO_NUM_TH | 1000 | 与旧测试相同定义的采样 PRO 阈值数 |

后台运行查看日志：

```bash
tail -f logs/ld_current_network_v1.log
```

结果集中在 `WORK_DIR`：

```text
config.json                 # checkpoint / 数据 / 预训练权重 / 代码哈希与完整配置
runtime.json                # 实际恢复的视觉层、HFA 层和 MARA 配置
predictions/*.npz           # 分类别概率图，留在服务器，不放入小报告包
prediction_lock.json        # 全部类别推理结束后的预测哈希
metrics.csv / metrics.json  # 分类别 Base / MARA 指标与有效类别宏平均
image_scores.csv            # 每张图显式标签、原始图像分数及固定阈值结果
REPORT.md                   # 可读结果汇总
visualizations/*/*.png      # 原图、框级 GT、固定 0..1 色标热图
run.log
ld_test_report.zip          # 小型结果包，可下载回来分析
```

图像标签独立读取，原始概率图不做逐图或逐类别 min-max。
Cam5 单类别 I-AUROC 为 `null` / 空 CSV / `N/A`，宏平均显示实际有效类别数（正常应为3），
另报告全部251张的 pooled image AUROC。定位指标仍是框级代理标注口径；
PRO 复用已验证的 `sampled_pro` 加速实现，与 `test2.py` 的采样阈值定义一致，并非另一实验的 exact PRO。
最佳像素 F1 是测试集描述统计，不是可部署阈值；背景 FPR@0.5 也只作诊断。

同一 `WORK_DIR`、数据、权重和配置再次运行会校验并复用已完成类别。
更改网络/权重、数据、图像尺寸、阈值或代码后会要求新输出目录；
批量和 worker 数可为显存/性能调整，旧预测不受影响。
服务器数据/权重路径允许重定位，但必须与保存的模块结构匹配。
本地测试覆盖真实 DataLoader、完整报告/续跑链路和真实小型 MARA agent 的前向委托；
没有在本机伪造完整模型性能，实际 251 张 GPU 推理需在服务器运行。

## 上传及安装（Linux 算力平台）

将 `LD_AD_20261116_bundle.tar.gz` 和同名 `.sha256` 文件上传到：

```text
/mnt/qfg/Tate_qfg/data/
```

在服务器终端执行：

```bash
mkdir -p /mnt/qfg/Tate_qfg/data
cd /mnt/qfg/Tate_qfg/data
sha256sum -c LD_AD_20261116_bundle.tar.gz.sha256
tar -xzf LD_AD_20261116_bundle.tar.gz

conda activate qfg_addino
python ld_adapter/install_ld_dataset.py --repo /mnt/qfg/Tate_qfg/SRA-DINO
cd /mnt/qfg/Tate_qfg/SRA-DINO
python tools/check_ld_dataset.py --data-root /mnt/qfg/Tate_qfg/data/LD_AD_20261116
```

解压后数据直接在 `data/LD_AD_20261116`，同层 `ld_adapter` 是小型代码适配包，
包含加载器、安装器、转换器、校验器和这份说明。第一次安装会备份服务器上的
`Datasets/__init__.py`，追加 `ld` 注册，保留其他数据集路径和所有测试脚本。
再次安装相同版本可重复运行；若服务器已有不同版本适配器，脚本会报错供检查，不直接覆盖。
使用项目现有训练环境即可，不需要 YOLO/ultralytics。

## 后续测试参数

先使用**在其他数据上事先选定的**权重，不要根据本测试集反复选模型或调整阈值。
`test.py/test2.py` 的 `--weight_path` 接收原有格式的 checkpoint 目录，
`test_mara.py` 支持指定一个 MARA checkpoint 文件。
不要向 Base 入口传入 MARA 权重，也不要把不匹配结构的 checkpoint 混用。
以下变量需要先设为你实际已有的目录/文件，不内置猜测的权重地址：

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
export LD_DATA_ROOT=/mnt/qfg/Tate_qfg/data/LD_AD_20261116

# Base：先设置 BASE_CKPT_DIR 为与此入口兼容的实际权重目录
: "${BASE_CKPT_DIR:?请先设置 BASE_CKPT_DIR 为实际的 Base checkpoint 目录}"
python test2.py --dataset ld --weight_path "$BASE_CKPT_DIR" \
  --result_path ./results/ld_base --device cuda:0 --batch_size 4 \
  --auto_hfa_from_ckpt --norm_mode none

# 或 MARA：先设置 MARA_CKPT 为事先选定的实际 MARA checkpoint 文件
: "${MARA_CKPT:?请先设置 MARA_CKPT 为实际的 MARA checkpoint 文件}"
python test_mara.py --dataset ld --weight_path "$MARA_CKPT" \
  --result_path ./results/ld_mara --device cuda:0 --batch_size 4 \
  --eval_latest_only --norm_mode none
```

原 `test.py` 同样增加 `--dataset ld` 即可调用数据接口，
但它只查找 `epoch_N.pth`，且是另一套固定结构的加载逻辑；
请选择与你现有权重兼容的入口。网络前向与实际性能仍需在服务器上用真实权重验证。

## 可重复转换与核验

```powershell
python tools/prepare_ld_dataset.py --source "D:\project\codex\LD_数据集资产_20261116" --output "D:\project\codex\SRA-DINO\data\LD_AD_20261116"
```

输出目录必须尚不存在，避免覆盖既有数据。核验已有转换结果：

```bash
python tools/prepare_ld_dataset.py --validate-only --output /mnt/qfg/Tate_qfg/data/LD_AD_20261116
python tools/check_ld_dataset.py --data-root /mnt/qfg/Tate_qfg/data/LD_AD_20261116 --full-batches --workers 2
```

核验覆盖所有图片解码、图像/标签/掩码 SHA256、真实图像尺寸、逐框掩码内容、
清单与文件一一对应，以及拍摄组不能跨 train/test。
转换时还检查所有数据的文件字节与解码 RGB 均无重复。
DataLoader 检查使用真实 PyTorch，对两个 split 的四类别逐一组批；
`--full-batches` 遍历全部图片，默认只检查每类每个 split 的正常/缺陷代表样本。
