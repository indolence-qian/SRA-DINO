# 从四数据集已有预测导出热力图

入口：`run_render_four_dataset_heatmaps.sh` / `render_four_dataset_heatmaps.py`。
适用于 `run_train_test_four_datasets.sh` 的已完成预测目录。
只需要 NumPy 和 Pillow，在CPU运行，不加载 DINO、CLIP、Flow 或文本头，不重新训练/推理。
脚本独立于封存训练代码，更新后可直接读取原来的预测。

旧输出 `panels/` 每类别仅抽两张256像素小图，且没有单独的彩色热力图。
这个入口默认处理四个数据集全部测试图，使用经典蓝—青—黄—红色图，并保持原图宽高比。

## 服务器运行

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main
conda activate qfg_addino
mkdir -p logs

nohup env WORK_DIR=./checkpoint/residual_hr_text_four_datasets_v1 \
  PACK=1 bash run_render_four_dataset_heatmaps.sh \
  > logs/four_dataset_heatmaps.log 2>&1 &
```

```bash
tail -f logs/four_dataset_heatmaps.log
```

默认自动选择 seed42 下数字最大的、已经生成 `locked.json` 的训练 horizon。
例如训练到15轮的输出为：

```text
checkpoint/residual_hr_text_four_datasets_v1/heatmaps/seed42/through15/source/
  index.html                          可下载到本机打开的图片浏览页
  index.csv / index.json              原图地址、标签、原始分数与渲染对应关系
  render_config.json                  色标/源阈值/预测锁及脚本哈希
  ld/gear_teeth/
    <id>_panel.jpg                    六联图
    <id>_full_heatmap.png             完整模型的纯热力图
    <id>_visual_heatmap.png           纯视觉分支热力图
    <id>_overlay.png                  完整模型热力图叠加原图
    <id>_prediction.png               原始分数按固定源阈值二值化
    <id>.json                        该样本的色标、分数范围和输出文件校验和
  ...                                其余类别及 MVTec/BTAD/MPDD
  heatmaps.zip                       PACK=1时生成，可整体下载
```

六联图上排依次是原图、GT红色标注、源阈值下的二值预测；
下排是纯视觉热图、完整模型热图、完整模型叠加图，并附低分到高分的色条和原始分数范围。
GT单独显示，不影响热图色标或二值预测。LD的GT是矩形框代理标注。
缩略GT按正覆盖保留微小前景；BTAD全零mask仍显示为无像素标注，图像异常标签独立保留。

`index.html` 使用相对路径。整体下载解压 `heatmaps.zip` 后打开，能直接浏览图片；
它不是 `four_datasets_report.zip`，新ZIP专门包含热图及索引，不包含模型或预测NPZ。

## 对比度与颜色解释

默认 `NORMALIZATION=source`：

```text
display = sigmoid((原始分数 - 该分支的固定源CAL阈值) / TEMPERATURE)
```

色标中点对应源阈值，各图使用同一规则。两个分支分别使用自身已固定的源阈值。
这些颜色是显示值，不能解释为异常概率；模型分数、指标和阈值都不修改。
图片仍比较暗时，可能是分数确实低于源阈值，可通过下面的显示增强查看局部响应：

```bash
WORK_DIR=./checkpoint/residual_hr_text_four_datasets_v1 \
  NORMALIZATION=percentile PACK=1 bash run_render_four_dataset_heatmaps.sh
```

增强版默认将**同一张图的 visual/full 两组分数联合的1%—99%分位数**映射到色条两端。
两分支共用显示范围，避免把弱文本增量放大成明显模型差异。
增强只用于看空间分布：**不同图片之间的颜色强度不可直接比较，红色不代表该位置超过实际检测阈值**。
二值预测图仍使用原始分数和固定源阈值。
增强结果独立保存到 `heatmaps/seed42/through15/percentile/`，不覆盖默认图。

## 只看LD或先少量预览

```bash
# 只输出LD的全部测试图
DATASETS=ld PACK=1 bash run_render_four_dataset_heatmaps.sh

# 四数据集每个类别先固定抽20张，不按预测好坏挑图
LIMIT_PER_CATEGORY=20 bash run_render_four_dataset_heatmaps.sh

# 只看轴面Cam5
DATASETS=ld CATEGORIES=shaft_cam5 bash run_render_four_dataset_heatmaps.sh
```

默认 `LIMIT_PER_CATEGORY=0` 代表全部图片。抽样按固定哈希，和GT/模型分数无关。
重复运行相同设置会校验并复用已完成图片；从少量预览扩展到全量时会补齐其余图片。
索引和可选ZIP包含本次选中的图片，其他已渲染文件保留。

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| WORK_DIR | `./checkpoint/residual_hr_text_four_datasets_v1` | 原四数据集实验目录 |
| TRAIN_SEED | 42 | 已完成的模型种子 |
| TRAIN_UNTIL_EPOCH | 自动 | 数字最大的已锁定训练horizon |
| DATASETS | mvtec,btad,mpdd,ld | 可选一个或多个测试集 |
| CATEGORIES | 全部 | 可选类别，逗号或空格分隔 |
| LIMIT_PER_CATEGORY | 0 | 0为全部；正数为每类别固定抽样数 |
| MAX_SIDE | 512 | 图片最长边；保持原始宽高比 |
| NORMALIZATION | source | source固定阈值色标，或percentile显示增强 |
| TEMPERATURE | 1 | source模式的显示过渡宽度，越小越陡 |
| LOW_PERCENTILE / HIGH_PERCENTILE | 1 / 99 | percentile显示范围 |
| OVERLAY_ALPHA | 0.45 | 热力图叠加比例 |
| SHOW_GT | 1 | 0可在没有GT文件时生成热图 |
| PACK | 0 | 1额外生成热图ZIP |
| OUTPUT_DIR | 自动 | 自定义热图目录；改显示参数需新目录 |

完整导出会生成较多图片；可先 `LIMIT_PER_CATEGORY=20` 检查效果。
修改色标、尺寸、透明度或脚本版本时，使用新的 `OUTPUT_DIR`，防止同一目录混入不同显示口径。
source与percentile模式默认目录已经分开。

## 输入完整性与常见提示

- 必须在原服务器上保留预测 `predictions/seed42/through15/*.npz`、sidecar JSON、
  `locked.json`、config/manifest、源选模JSON、`sealed_eval.json` 以及原图。
  只有小型分析ZIP时无法重建模型热图，需要在服务器原工作目录运行。
- 没有已完成预测锁时脚本会明确报错，不能在仅完成训练、尚未完成预测时伪造热图。
- 脚本核验清单、源选模、预测锁、每个渲染样本的NPZ及sidecar、原图校验和；
  若已有评价记录，也会核验对应GT。它不会更改这些输入、封存代码或历史指标。
- 原图需要与预测时一致。改动GT后已有图片会拒绝静默复用。
- 输出是现有评分图的插值和着色，不增加网络预测分辨率，也不会改善模型指标。
