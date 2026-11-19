# 四数据集 HTML 交互分析报告

当前实验：`checkpoint/residual_hr_text_four_datasets_v1`，即你已运行的
`run_train_test_four_datasets.sh`。新增报告读取该实验源选模后权重生成的锁定预测，
不训练新模型，也不更改权重、封存预测或历史评价。

## 一体化入口：先可视化当前最新实验

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main
conda activate qfg_addino
mkdir -p logs

nohup env PIPELINE_MODE=visualize SAMPLE_FRACTION=0.1 \
  WORK_DIR=./checkpoint/residual_hr_text_four_datasets_v1 \
  bash run_four_datasets_with_html.sh \
  > logs/four_dataset_dashboard.log 2>&1 &
```

```bash
tail -f logs/four_dataset_dashboard.log
```

上面的命令严格只读取已有锁定预测，不训练、不补推理。默认CPU运行，自动选择 seed42 下最新已完成并锁定的 horizon。
统一入口默认为 `auto`：已有锁定预测就直接补报告；已有实验但缺少预测时，
用该实验已训练权重恢复预测/评价后补报告，不训练新头。
若默认v1目录不存在，会报错而不会意外开始新的训练。
独立的 `run_build_four_dataset_dashboard.sh` 仍保留用于只执行报告阶段。
若完整运行15轮，结果在：

```text
checkpoint/residual_hr_text_four_datasets_v1/analysis/seed42/through15/source_sample10_v2/
  index.html                交互式分析首页
  HTML_ANALYSIS.html        包内入口，仅相对跳转到 index.html
  analysis.json             全量数据集/类别指标、抽样逐图诊断与来源哈希
  sample_metrics.csv        仅抽样图片的逐图数据，含两个分支指标
  sampling.json             每层总体/抽样数、采样方法和选中ID
  asset_manifest.json       包内资源相对路径及SHA256
  README.txt                解压、离线浏览和静态托管说明
  gallery.html              同样可按数据集切换的简单热图页
  mvtec/...                 原图缩略图、GT、误差图、热图、叠加图等
  btad/...
  mpdd/...
  ld/...
  analysis_report.zip       HTML和本次抽样的全部资源，可整体下载
```

每次成功生成报告还会更新固定入口：

```text
checkpoint/residual_hr_text_four_datasets_v1/reports/HTML_ANALYSIS.html
checkpoint/residual_hr_text_four_datasets_v1/reports/html_analysis.json
```

固定HTML入口会跳转到本次生成报告；JSON记录具体seed、horizon、路径和预测锁哈希。

整体下载并完整解压 `analysis_report.zip`，双击 `index.html` 或包内的 `HTML_ANALYSIS.html` 即可离线浏览，无需启动服务器、
安装额外浏览器插件或连接外部网络。单独拷贝HTML会丢失图片，需保留相对目录或下载整个ZIP。
不要把服务器 `reports/HTML_ANALYSIS.html` 单独下载：它跳转到服务器上的相邻 analysis 目录；
包内入口已改为只引用包内资源。页面显示的原始服务器路径仅为溯源文字，不用于加载图片。
如需公网访问，将**整个解压目录**上传到支持静态网站的托管服务，访问 `index.html`；
下载ZIP并不会自动产生公网链接。

## 默认均匀抽取1/10图片

当前及后续训练的统一入口默认 `SAMPLE_FRACTION=0.1`。每个“数据集 × 类别 × 正常/异常图像标签”
组按文件名/ID排序，把组分成等宽区间并取中点样本；取 `ceil(组图片数 × 0.1)` 张。
因此大组约10%，小组至少一张、数量向上取整，实际总比例可能略高于10%。
不按GT前景大小、置信度或检测好坏挑选；同样的数据和参数会得到相同样本。
异常图即使mask全零仍属于异常组。`sampling.json` 可检查各层数量和具体ID。

**指标与图片范围分开**：数据集图表、类别表与混淆计数使用筛选范围内的全部测试图片；
逐图卡片、好坏筛选、页首抽样KPI及 `sample_metrics.csv` 只覆盖抽样图片，并在页面标明。
脚本仍在CPU读取全量锁定分数和GT以计算诊断统计，但只为1/10图片生成热图和对照资源。
原评价JSON存在时，核验预测锁和全类别GT哈希后复用其全量指标，不改写历史评价。

新版默认输出到 `source_sample10_v2`，不会混入旧 `source/` 全量图片。
ZIP按照本次明确列出的资源打包，不会把目录里遗留的其他图片带进去；
打包前检查资源存在且位于报告目录内，并写入校验清单。
需要全量图库时可显式设置 `SAMPLE_FRACTION=1`，会输出到独立的 `source_sample100_v2`。

如果只完成训练而尚无锁定预测，可显式要求用当前实验已保存的权重补齐预测：

```bash
RESUME_PREDICTIONS=1 WORK_DIR=./checkpoint/residual_hr_text_four_datasets_v1 \
  bash run_build_four_dataset_dashboard.sh
```

该选项调用原实验入口并设置 `RUN_TRAIN=0`，复用已有头权重和正常参考，不训练新头；
若特征不存在，会调用原冻结骨干重新导出。GPU、数据、预训练权重和原始配置仍需可用，
且必须与原运行相同。没有已训练快照时明确报错，不会悄悄开始一轮新训练。

## 页面可以分析什么

- **数据集概览**：MVTec/BTAD/MPDD/LD 的图像AUROC/AP、像素AUROC/AP、AUPRO、最佳像素F1、
  定位Dice与TP/TN/FP/FN诊断。宏平均和有效类别数明确显示，缺失指标为N/A。
- **类别明细**：按类别查看问题规模，点击类别后定位到对应样本；不把不同数据集直接混成一个分数。
- **分数据集图库**：热图区域有独立数据集下拉框，默认MVTec；切换BTAD/MPDD/LD时只显示对应数据集。普通热图浏览页也有相同切换，不再混排。
- **好坏样本筛选**：在当前数据集抽样内查看正常正确、异常检出且定位好、正常误报、异常未触发、图像检出但定位差、空像素标注。
- **排序**：定位Dice从低到高/高到低、背景误报面积、原始分数相对源阈值、文本带来的Dice变化。
- **文本分支比较**：同图查看纯视觉与完整模型，筛选文本改善/退化样本。
- **逐图放大**：原图、GT、完整模型/纯视觉热图、叠加图、二值预测及六联图；可用方向键前后浏览。
- **误差叠加**：绿色表示预测与GT命中，红色表示背景误报，蓝色表示标注区域漏检。
- **导出**：页面按钮可下载当前数据集筛选样本CSV；结果目录也包含本次抽样的逐图CSV。

页面会给出需人工核查的线索，例如微小标注区域召回低、背景响应扩散、检出但定位重叠低；
不会仅凭热图直接断言反光、油污、磨损等具体原因。应结合原图和拍摄情况进一步判断。

## 怎么定义“好”和“差”

图像诊断定义：`最大像素分数 >= 该分支的固定源CAL像素阈值` 视为触发。
这个像素阈值未单独按图像误报率校准，所以这里的TP/FP/FN/TN用于复盘，
不等同于部署图像分类阈值下的性能。图像AUROC/AP独立按连续分数和显式图像标签计算。

定位IoU/Dice/召回率使用与原评价一致的GT前景保留缩放，在原预测分辨率计算，
不在缩略图上打分。定位指标仅用于“异常且像素标注非空”的图片：

- **表现好**：正常未触发，或异常检出且定位Dice≥0.7。
- **问题样本**：正常误报、异常未触发，或定位Dice<0.3。
- **定位差**：异常有非零像素标注，Dice<0.3，即便图像已检出也单独标出。
- **文本改善/退化**：Dice变化超过±0.05，或完整模型相对视觉分支修复/引入图像诊断错误。

上述阈值只是复盘分组，不参与选模或更新权重。页面始终显示原始指标供你自行判断。
两个分支分别使用原来固定的源阈值，触发结果的变化不能单独证明文本抑制了视觉误报。

BTAD异常图的官方全零mask不丢弃，图像标签仍为异常，定位指标为N/A。
LD标注来自矩形框，框级定位Dice/AP不能当作精确分割指标。
Cam5图像标签只有异常时，类别图像AUROC为N/A，不当作0计入宏平均。

原评价文件与预测锁、全类别GT一致时，页面读取原逐类别全量图像AUROC/AP和像素AUROC/AP/AUPRO/F1；
图库只取1/10不会改变这些全量指标。图表/类别表明确标为全量评价，逐图统计明确标为抽样。
若原评价未完成，仍可浏览已锁定预测的逐图诊断，缺失的原像素指标显示N/A。

## 以后默认带HTML的运行方式

保留原训练脚本和封存哈希，以免破坏当前实验的续跑；新的推荐入口在它结束后自动追加HTML：

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
conda activate qfg_addino
mkdir -p logs

nohup env PIPELINE_MODE=train GPU_IDS=0,1 EPOCHS=15 \
  WORK_DIR=./checkpoint/residual_hr_text_four_datasets_v2 \
  bash run_four_datasets_with_html.sh \
  > logs/residual_hr_text_four_datasets_v2.log 2>&1 &
```

这是一个新的完整训练实验，所以例子使用新的v2目录。当前v1只需运行本页第一段补报告命令。
`train` 模式不指定 `WORK_DIR` 时，会自动创建带时间戳的新目录。

| PIPELINE_MODE | 行为 |
|---|---|
| auto（默认） | 分析指定的已有实验；有预测则直接生成HTML，缺预测则复用已有权重补推理；不自动启动新训练 |
| visualize | 只生成HTML，要求已有完成并锁定的预测 |
| resume | 明确用原实验权重恢复预测和评价后生成HTML，不训练新头 |
| train | 运行训练/评价，成功后自动生成HTML；原训练入口的断点规则继续生效 |

原 `RUN_EXPERIMENT=0/1` 的显式命令仍兼容，分别对应 `visualize/train`；
两者同时指定时，以明确设置的 `PIPELINE_MODE` 为准。
训练或恢复失败会停止流程，不会误报HTML生成成功。
当前和后续实验的HTML输出要求已写入仓库 `AGENTS.md`，后续新增实验入口也应提供同样的分析产物。

其他参数沿用热图导出：`DATASETS=ld`、`LIMIT_PER_CATEGORY=20`、
`NORMALIZATION=percentile`、`MAX_SIDE=512`、`OUTPUT_DIR=...` 等。
默认四数据集全量指标、均匀1/10图片、source固定色标、自动打包。
`LIMIT_PER_CATEGORY` 是抽样后的可选每类别数量上限，0表示不再额外限制；不代表取消1/10抽样。
percentile仅增强显示，不改变排名、IoU/Dice、误报/漏报或模型指标。

报告在本地用具有明确TP/TN/FP/FN、定位正确/错误与空标注的小样本验证统计，
并在真实浏览器检查筛选、排序和逐图查看。真实实验效果由你的v1缓存生成，不预先编造数值。
