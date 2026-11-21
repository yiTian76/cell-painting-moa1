# 利用细胞形态预测药物作用机制

本项目使用细胞形态特征预测药物作用机制。主任务是在 A549 细胞接受 48 小时药物处理后，从 79 种已知机制中预测一种未见药物的标签。模型输入是 Broad 已发布的 CellProfiler 形态测量。两批数据和两种剂量的处理后文件都在包内。训练和预测都可在 CPU 上离线完成，无需 OpenAI API Key。

## 主要结果

采用药物级分层五折，汇总每种药物唯一一次折外预测。主方法按七个特征组等权计算类内平均余弦。

| 方法 | 首选命中率 | 前三命中率 | 平衡准确率 | 宏 F1 |
| --- | ---: | ---: | ---: | ---: |
| 通道等权平均余弦（主方法） | 10.66% | 17.25% | 12.08% | 0.0764 |
| 多数类／频数排序 | 4.91% | 13.32% | 1.27% | 0.0012 |
| 全特征平均余弦 | 10.10% | 17.25% | 11.42% | 0.0743 |
| 余弦最近邻 | 16.97% | 25.11% | 16.43% | 0.1598 |
| 类别平衡岭分类器 | 16.55% | 23.70% | 17.94% | 0.1601 |

最近邻的首选命中率最高，岭分类器的平衡准确率与宏 F1 最高。当前结果未证明主方法优于全部对照。

## 已训练的模型在哪里

`models/channel_balanced.npz` 是本项目的主模型。它保存了从 713 种训练药物估计的标准化参数，以及 79 类机制在 7 组形态特征上的代表向量。`models/ridge_balanced.npz` 是岭分类对照，保存了每类的线性权重和截距。两个模型均记录了类别名称、特征顺序和训练药物的结构标识。

这两个文件供新药推理使用，均在五折评估完成后用第一批全部 713 种合格药物重新拟合。报告中的命中率来自五个分别训练、分别留出测试药物的模型。安装下文的依赖后，可直接加载已训练模型，预测过程无需读取训练表。

```powershell
.\.venv\Scripts\python.exe src/predict_saved.py --model models/channel_balanced.npz --query demo/new_profiles.csv.gz --out rerun/main_predictions.csv
.\.venv\Scripts\python.exe src/predict_saved.py --model models/ridge_balanced.npz --query demo/new_profiles.csv.gz --out rerun/ridge_predictions.csv
```

如需从相同训练数据重建两个参数文件，可运行以下命令。

```powershell
.\.venv\Scripts\python.exe src/export_models.py --train data/profiles_10uM.csv.gz --folds results/folds.csv --out rerun/models
```

## 准备小组汇报

- 实验报告提供 [PDF](reports/实验报告.pdf) 和 [可编辑 Word](reports/实验报告.docx)。共 9 页，包含五种方法的数学原理、真实药物计算例子、评价指标、形态解释及稳健性验证。
- 做 baseline 的同学使用 `data/profiles_10uM.csv.gz`，按 `compound_id` 与 `results/folds.csv` 做一对一内连接。划分文件内的 713 种药物就是评估全集。每次选择一个 `fold` 作测试，其余四折训练，每种药物仅评估一次。数据文件中另外 564 种药物所属类别没有达到 5 种结构的门槛。
- 只允许以 `Cells_`、`Cytoplasm_` 或 `Nuclei_` 开头的 1,609 项测量作为形态模型输入。`moa` 是真实标签。`compound_name`、`compound_id` 和所有 `Metadata_` 列用于核对身份或审计。任何需拟合的变换应只用该折训练数据。
- 对照汇总在 `results/all_model_metrics.csv`。对全部折外预测统一计算首选命中率、前三命中率、平衡准确率和宏 F1。根据不同方法的同类药物得分去重后选前三个不同机制。按每类等权计算宏 F1，对零命中的类别保留零分。

## 运行实验

在解压后的项目目录打开终端。已验证的实验环境为 Python 3.13.14。首次安装 Python 依赖需要联网，数据已经随包提供。以下命令适用于装有 Python 3.13 的 Windows PowerShell。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe reproduce.py
```

macOS 或 Linux 可用 `python3.13 -m venv .venv` 创建环境，后两行使用 `.venv/bin/python`。中文图表默认搜索微软雅黑、思源黑体等字体。指定其他中文字库时可运行 `.venv/bin/python reproduce.py --font /path/to/CJK-font.ttf`。

程序依次做五折训练、499 次标签置换、1,000 次分层重抽样、特征组消融、岭分类对照、低剂量测试、第二批验证、板布局诊断、无标签预测演示与绘图。新结果写入 `rerun/`，并自动与随包的划分、结果表和演示预测核对。完整五折汇总在 `results/all_model_metrics.csv`，每折与分类别的明细均保留在 `results/`。

## 单独演示新药预测

```powershell
.\.venv\Scripts\python.exe src/predict_new.py --train data/profiles_10uM.csv.gz --query demo/new_profiles.csv.gz --out rerun/demo_predictions.csv
```

`demo/new_profiles.csv.gz` 是第二批次两种真实药物的形态值，已移除机制标签。示例用于说明输出如何阅读。这两种药都没有进入第一批训练队列。曲美替尼（trametinib）的首选为 MEK 抑制剂。伏立诺他（vorinostat）的首选为蛋白酶体抑制剂，其已知 HDAC 抑制机制排在第三。全部 24 种新结构的表现见批次验证表，示例的命中情况仅代表这两种药。输出分数表示平均余弦相似度，不能解释为概率。

替换查询文件时，每种化合物提供一行、相同的特征列、匹配的归一化口径和 `compound_id`（InChIKey 或其前 14 位）。可选列是 `compound_name` 和用于事后核对的 `moa`。工具默认检查训练与查询结构重叠。

## 文件索引

| 文件 | 用途 |
| --- | --- |
| `data/profiles_10uM.csv.gz`、`data/profiles_3p33uM.csv.gz` | 第一批 1,277 种单标签结构在两个精确剂量下的独立共识谱 |
| `data/profiles_batch2_10uM.csv.gz` | 第二批 199 种单标签结构，其中 141 种属于 79 类候选机制 |
| `data/data_provenance.json`、`data/batch2_provenance.json` | 源文件 URL、固定版本、SHA-256、纳排计数与处理后文件校验值 |
| `data/sample_flow.csv`、`data/batch2_annotation_flow.csv`、`data/feature_manifest.csv` | 药物及特征的纳排审计，第二批标签通过 Broad 目录映射与核对 |
| `results/folds.csv`、`results/oof_predictions.csv` | 五折划分及逐药预测，含名称、真值、主模型前三候选与分数 |
| `results/per_class.csv`、`results/confusion_main.csv` | 各类样本数、预测数、正确数与精确率、召回率、F1；主模型 79 × 79 混淆矩阵 |
| `results/ablation.csv`、`results/feature_evidence.csv` | 逐组移除后的结果，以及各机制真实类别相对最强竞争类别的特征得分贡献 |
| `results/class_readouts.csv` | 六项易解释形态测量在各类中的中位数、四分位数及其余药物的中位数 |
| `results/supplement_predictions.csv`、`results/batch2_predictions.csv` | 低剂量、岭分类器及第二批次的逐药预测 |
| `results/layout_metrics.csv`、`results/leave_layout_out_metrics.csv` | 只看板位与按板布局整组留出的诊断结果 |
| `results/bootstrap_ci.csv`、`results/permutation_scores.csv`、`results/run.json` | 对固定折外预测的区间、置换对照、随机种子、环境与运行记录 |
| `figures/`、`src/` | 可放入汇报的中文 PNG 图，以及所有数据处理、建模、验证和绘图脚本 |

## 从公开源文件重建输入

常规复现直接使用 `data/`。如需从 Broad 的固定仓库版本重建三个数据文件，可在已安装依赖的环境运行以下命令，下载的源文件缓存在 `raw_data/`。

```powershell
.\.venv\Scripts\python.exe src/prepare_data.py --cache raw_data --output rebuilt_data
.\.venv\Scripts\python.exe src/cross_batch.py prepare --primary rebuilt_data/profiles_10uM.csv.gz --cache raw_data --out rebuilt_data
```

资料提供方将数据、结果与图表以 CC0 1.0 发布，随包附 `data/LICENSE_CC0.md`。上游试剂量字段名沿用源仓库，报告中的 µM 按原论文实验浓度解释。发布信息、原始研究与染色通道说明分别见 [Natoli 等的公开数据](https://doi.org/10.5281/zenodo.5008187)、[Way 等的研究](https://doi.org/10.1016/j.cels.2022.10.001) 和 [Bray 等的 Cell Painting 方法](https://doi.org/10.1038/nprot.2016.105)。
