# Medium decision explanations

## 阅读入口

- `thesis_explanations.md`：方法说明，以及全部 30 个结果的简短英文解释和对应文献。
- `decision_explanations.csv`：全部决策、时间和英文理由，便于筛选。
- `cross_formation_comparisons.md`：全部 30 个决策分别与其余四种 formation 对比，包含五个位置的耗电率、总时间和逐组机理解释。
- `cross_formation_best_available_spacing.csv`：120 组比较，各个其他 formation 使用自身最优的可用间距。
- `cross_formation_matched_spacing.csv`：99 组相同间距的比较；不补造缺失或安全排除的数据。
- `artifact.json`：完整分析报告的规范数据；已通过分析报告校验及渲染工具。
- `literature_sources.json` / `references.bib`：10 篇原始研究的来源、适用范围和引用信息。
- `method_notes.md`：各个结果的重复实验检查，独立于正文的机理解释。
- `numeric_audit.json` / `audit_summary.json`：精确排程复核与稳健性检查。

论文用 **low wind = 原 Lv1**，**high wind = 原 Lv2**。这是实验风扇档位标签，不能当成风速测量。电量区间仍称 Medium SOC。论文拼写使用 echelon；历史代码的 `echalon` 不更名。

范围为已有 250 cm 前进实验的 Medium 真实耗电率。统一到 Bideal 75% 是前置处理，函数不处理 SOC。输出对应固定 25 s 飞行及随后全队完成充电的计算时间，不是任意实时 SOC 或整个多段任务的全局最优。没有加入 wind tunnel 模拟数据，没有改变原始实验和此前冻结的决策。

## 函数调用

```python
from ml_policy.medium_decision_explanations import select_medium_with_explanation

result = select_medium_with_explanation(2, "head", "low")
print(result["configuration"])        # front, 75 cm, 原位置分配
print(result["total_time_minutes"])   # 190.766...
print(result["data_explanation_en"])
print(result["aerodynamic_interpretation_en"])
print(result["caveats"])
```

输入仍然只有 charging pad availability、wind direction、wind strength 三项。附加的 `references` 给出文献及其适用边界，`sensitivity` 给出对重复实验波动的敏感性。原函数继续兼容数值风级；这个新增入口使用 low/high。

完整命令行输出：

```sh
python3 -m ml_policy.medium_decision_explanations --pads 2 --wind head --wind-strength low
```

## 解释应如何使用

正文解释“队形与风向如何改变来流、尾流和抗风推力需求”，然后连接到“哪些位置形成充电瓶颈、总时间为何较短”。新增的旋翼风洞文献给出了串列下洗流、侧向错位和推力/功率之间的具体联系。每条英文理由不再反复插入统计说明，重复实验检查集中在方法备注中。气动机理用于解释，耗电率和决策时间仍取自原数值分析；二者不混写成额外测量。

120 组跨 formation 对比中，112 组的选择具有更低的全队标准化 SOC 消耗；另外 8 组虽然总时间更短，但全队标准化 SOC 消耗更高。这 8 组的原因在于充电任务的分布和瓶颈，而不是“整体更省电”。比较表保留五个位置的耗电率以及最后完成的充电位。跨队形的 P1–P5 是各队形记录中的位置编号，不假定同一个编号具有相同的气动角色。

## 论文图

- `output/pdf/medium_configuration_decision_map.pdf`：矢量 PDF，180 × 111 mm，适合双栏通栏。
- `output/figures/medium_configuration_decision_map.svg`：保留可编辑文字的矢量图。
- `output/figures/medium_configuration_decision_map.png`：600 dpi 位图。
- `output/figures/medium_configuration_decision_map.tex`：英文图注及 LaTeX figure* 插入片段。
- `output/figures/medium_configuration_decision_map_data.csv`：图中 30 个格子的底层数据。

图中 low/high 指风力，所有电池都处于前置处理后统一的 Medium baseline。每个格子的文字标明 formation 和 spacing，黑白打印不依赖颜色区分。已对照冻结结果核对全部格子，并检查 PDF 渲染及 PNG。

## 可复核性

原冻结文件的 SHA-256 保留在 `manifest.json` 中。新增查询函数会拒绝与冻结决策不一致的解释。全部 275 个候选排程经过精确复核，30 个输出保持一致。数值审计程序为 `output_py/interpret_medium_decisions.py`；报告整理程序为 `output_py/package_medium_interpretation.py`。SQL 文件记录报告表格的实际整理步骤；重采样和排程在 Python 中进行，不伪装为 SQL 计算。
