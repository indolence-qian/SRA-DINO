# P2：语义质量 → 空间匹配 → 小缺陷监督

本轮回答三个问题：局部描述是否贴合图像；冻结视觉特征与这些文字是否存在空间匹配信号；显式匹配和小缺陷监督能否带来独立收益。它是上游定位头实验，尚未加入 MARA。

## 为什么继续做这三步

上一轮语义审计证明文本改变会影响原始残差和部分最终像素，因此不能说语义完全没进入流程。但真实文本与打乱文本没有稳定的额外收益，源域背景压低、目标域背景抬高的现象也仍存在。固定 0.5 阈值的小缺陷命中数上涨，并不等于控制相同背景误报率后定位更好。

本轮不把“解析成功”当成语义正确，不把余弦相似度直接当成缺陷概率，也不把 alpha=0 回退计为改进收益。

## 1. 源域语义质量复查

`quality/tiles.csv`：全部源 train/val 局部文本、状态、独立角色有效性、词数、两种文本的余弦、图像级截断数量，以及源域 GT 的局部覆盖诊断。

`quality/summary.csv`：按类别统计弃权/解析、重复描述、observation 与 normal expectation 过于相似、文本截断情况。重复正常预期可能合理；这些统计只能提示检查方向，不能自动判定描述错误。

`quality/index.html`：默认每个源类别导出 6 个样本，按源域小缺陷、正常图、其他缺陷、异常图的背景局部及回答状态分层，层内按固定种子排序。优先复制 trace 中实际 `post_vision_*.png` 输入并校验 SHA；缺少这些文件时重建输入，并明确标注。另附源 GT 裁剪供人工核对，GT 不进入 VLM 或定位头输入。

`quality/manual_review.csv`：记录局部缺陷是否可见、描述是否贴合、正常预期是否有用和备注。续跑不会覆盖人工记录。自动阶段只导出复查材料；没有人工填写时，`human_accuracy=NOT_MEASURED`，不会凭空报告语义准确率。人工记录不自动转成训练标签，不强制改变合理弃权。

旧缓存只包含图像级 CLIP 截断计数，因此不声称已定位哪一条文本被截断。

## 2. 显式空间匹配与源域诊断

使用缓存中冻结适配后的 DINO 特征和冻结 CLIP 文本编码。在它们共同的维度空间做归一化余弦，位置匹配发生在定位头 hidden 投影之前。每个视觉层分别产生：

- observation 匹配图；
- normal expectation 匹配图；
- 同一 tile 的 observation − expectation 匹配图。

匹配只在 tile 覆盖的特征网格内生效；重叠 tile 平均；两个角色分别保持有效性。差值要求同一 tile 两种角色都有效，不能拿两个不相关 tile 的角色拼成差值。差值只作为头的输入，正号不自动代表异常。

`matching/metrics.csv`：只用源 val，在固定、与标签无关的像素采样位置上，报告真实/打乱文本的原始匹配 AUC、GT/背景均值及有效像素数。它不是最终异常检测分数，缺少正例或背景时 AUC 留空。`matching/panels/` 导出按类别顺序选择的源域样本匹配图。

`matching/comparisons.csv`：逐类别、角色、视觉层计算真实与打乱文本的匹配 AUC 差异和 GT/背景分离情况。

`matching/shuffle_audit.csv` 和 `donors.csv`：打乱文本来自其他图像，严格限制在同一 partition/dataset/category，优先同一 tile 和同一角色。原有 coverage、box、缺失角色不变，不使用标签或 Base 分数选择 donor。缺少 donor、相同文本造成的无效打乱都明确计数。

当前特征仍是原始缓存的网格，例如 512 输入对应 32×32 特征。插值用于对齐显示/损失，不增加真实的空间细节。如果语义质量或匹配信号不足，应先据这些结果决定是否重建更清晰的局部输入或更细特征缓存。

## 3. 两套损失、三种训练对照

| 组别 | 空间匹配头 | 损失 | 训练文本 |
|---|---|---|---|
| p2a/visual | 相同结构，语义通道关闭 | P1 | 无 |
| p2a/real | 相同结构 | P1 | 真实 |
| p2a/shuffled | 相同结构 | P1 | 同源 partition/category 的其他图像文本 |
| p2b/visual | 相同结构，语义通道关闭 | P1 + 连通区域 BCE + 局部排序 | 无 |
| p2b/real | 相同结构 | 同上 | 真实 |
| p2b/shuffled | 相同结构 | 同上 | 打乱 |

六组从相同随机种子初始化，使用相同训练轮数、优化器、样本顺序和源验证规则。它们各自训练；推理时关闭一个已有头的语义分支，不等价于这里的 visual 训练组。

先看 p2a/real 相对 p2a/visual、p2a/shuffled 的差异，判断真实语义是否提供额外收益；再看各模式 p2b 相对 p2a 的变化，判断小缺陷损失的作用。P2 的 visual 结构与旧 P1 并不完全相同，不能把跨版本差异全部归因于空间匹配。

P2b 将每个保留的 GT 连通区域分别计算正例 BCE，然后按区域加权平均，避免一个大缺陷仅因像素多就压过所有小区域。面积不超过图像 0.1% 的区域，默认相对权重 2；默认忽略少于 2 像素的区域，仅不参与新增区域项，仍参加原全图损失。区域周围默认 8 像素范围内，选择至多 64 个高分背景像素，约束缺陷与邻近背景的 logit 排序。其他 GT 从背景集合剔除，正常图新增项为零。保留 P1 背景抬升惩罚和残差约束。

默认新增区域项、排序项各权重 0.1。排序在 logit 空间计算，极低 Base 分数仍有优化梯度。它不是低分必然能修复的保证。

## 划分、选择和结果解释

沿用封存的 source train/val/target 划分。源数据 TEST 掩码用于训练新的定位头；原 Base 可能看过源 val，因此不要声称这是完全未见的源验证。目标标签只用于最终报告，不参加训练、epoch/alpha 选择或部署阈值选择。

保留 P0 精确概率残差与 P1 源验证门控，包括 alpha=0 的 Base 回退。`best.pt` 是源验证选中的结果；`last.pt` 使用 best 的源选择 alpha，另报告 alpha=1 的诊断。某组 best 回退后，last 的原始残差及 alpha=1 图仍可判断模型有没有学习，不能误认为整个训练从未参与。

`results/metrics.csv` 分源 val/目标 eval、类别、损失、训练模式、best/last 和推理干预报告 PRO、像素 AUROC、背景误报与小缺陷指标。所有匹配 FPR 指标明确标为 DIAGNOSTIC：目标数据自身的标签用于计算这个曲线诊断，阈值不得拿来部署。

`results/interventions.csv` 记录真实文本头在真实/打乱/关闭分支下的原始残差差异和最终概率差异。它与独立训练的三组对照互补。仅单个种子的微小差异不支持统计显著性结论。

`results/comparisons.csv` 给出各头相对 Base、真实语义相对 visual/shuffled、P2b 相对 P2a 的配对差值，单位为百分点。`results/README.md` 汇总 best 的类别宏平均，方便先看整体趋势。

## 运行

不重跑 Qwen，不加载 DINO/CLIP，不需要启动 VLM 环境。`SOURCE_WORK_DIR` 必须是**含 config.json、manifest.json、sealed.json、features、reviews、traces 的原始定位缓存目录**，不是 P0/P1 或 semantic audit 的结果目录。

在训练环境执行：

```bash
conda activate qfg_addino
cd /mnt/qfg/Tate_qfg/SRA-DINO

nohup env \
  TRAIN_PYTHON="$(command -v python)" \
  SOURCE_WORK_DIR=./checkpoint/upstream_localization_v1 \
  WORK_DIR=./checkpoint/upstream_P2_v1 \
  GPU_IDS=0,1 \
  HEAD_EPOCHS=15 \
  HEAD_BATCH_SIZE=4 \
  bash run_exp_upstream_p2.sh \
  > nohup_upstream_P2_v1.out 2>&1 &
```

脚本按质量导出 → 匹配诊断 → 六组双卡 DDP 训练 → 双卡按类别分片评估 → 汇总顺序运行，前序失败停止后续。重复同一配置自动续跑已存的训练。输出目录加锁；修改代码或配置必须换 `WORK_DIR`，防止混用实验。

如先只收集质量/匹配证据，在上述 `env` 中加入 `RUN_TRAIN=0 RUN_EVAL=0`，其余参数保持一致；之后恢复默认即可训练。人工复查材料不会自动触发暂停或批准流程。

默认全部结果在服务器 `/mnt/qfg/Tate_qfg/SRA-DINO/checkpoint/upstream_P2_v1/`：质量材料在 `quality/`，匹配结果在 `matching/`，检查点/训练日志在 `heads/{p2a,p2b}/{visual,real,shuffled}/`，总表在 `results/metrics.csv`。外层日志是项目目录的 `nohup_upstream_P2_v1.out`。

## 本地验证

`tests/test_upstream_p2.py` 使用合成 CPU 缓存，验证匹配支持与角色缺失、精确 Base 身份、语义梯度、小区域梯度与邻近背景排序、源划分隔离、只读缓存、训练对照、断点续跑、Base 回退和 Bash 阶段顺序。完整真实数据实验需在服务器运行后判断是否涨点。
