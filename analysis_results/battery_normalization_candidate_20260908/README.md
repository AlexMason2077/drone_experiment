# 五块电池独立三段曲线 + 三段 Bideal（候选版）

状态：离线候选校准工具已建立；未替换旧版耗电率表、未接入训练或飞行控制，未改动原始数据。

## 数据与拟合

- 使用 2026-09-06 五条真实单机悬停 baseline。用户确认均无风、相同设定高度。
- 配对：B10/D2、B11/D1、B12/D5、B13/D3、B14/D4。B12/D2 的 20260906_165812 不参与。
- 共同校准范围为 95%→20%；当前候选不外推到 100% 或低于 20%。
- 每块电池独立搜索整数 SOC 分界点。每段至少 10 个 SOC 百分点、20 秒；这是防止过短段的建模约束，不是电池物理定律。
- 线段连接真实首次达到 95%、两处分界点、20% 的采样点；用范围内原始采样 SOC 的 RMSE 选择分界。不是无约束直线回归。
- 三段保证连续、下降；分界点是本次数据的经验近似，不等同于已验证的电化学阶段。

| 电池 | 三段 SOC 范围 | 三段耗电率（SOC 百分点/分钟） | 拟合 RMSE（SOC 百分点） |
|---|---|---|---|
| B10 | 95% → 63% → 43% → 20% | 7.351 / 9.791 / 12.378 | 1.039 |
| B11 | 95% → 67% → 47% → 20% | 7.368 / 9.335 / 13.205 | 0.971 |
| B12 | 95% → 79% → 48% → 20% | 11.243 / 9.643 / 14.897 | 1.127 |
| B13 | 95% → 81% → 51% → 20% | 30.962 / 6.973 / 13.383 | 1.563 |
| B14 | 95% → 85% → 56% → 20% | 29.870 / 6.464 / 12.877 | 1.970 |

## Bideal 的定义

1. 在相同 SOC 下，取五块电池各自曲线的耗电率，各占 1/5 权重：b_mean(S) = Σ b_i(S)/5。
2. 对 60/b_mean(S) 按 SOC 积分，得到共同参考的 SOC–时间曲线。由于个体分界不同，这时不止三段。
3. 将该参考再次近似为三条连续线段。整数 SOC 分界与最短段限制同上；按整个时间轴的精确积分平方误差选取。

这不是直接平均各电池第一段、第二段、第三段的斜率，也不是平均整段飞行时间。

Bideal 范围：95% → 83% → 51% → 20%。
Bideal 耗电率：16.352349 / 8.321562 / 12.968637 SOC 百分点/分钟。
参考近似 RMSE：0.4904 SOC 百分点，最大偏差 0.8200。
该误差比较的是两条构造曲线，并不是独立实验预测误差。

## 如何换算真实实验区间

对于电池 i 在实际 Δt 秒内从 S_start 降到 S_end：

- H = ∫[S_end,S_start] 60/b_i(S) dS：对应自身 baseline 的等效悬停秒数。跨个体分界点分别积分。
- g = H/Δt：相对于自身 baseline 的 SOC 耗电因子。不是测得的功率比。
- 明确选择共同参考起始 SOC q（例如 70%），让 Bideal 前进 H 秒，得到 q_end。跨 Bideal 分界点分别处理。
- 标准化耗电量 = q−q_end；标准化平均耗电率 = 60(q−q_end)/Δt。

实际 SOC 原值保留，不乘比例；新字段以 bn_ 开头。SOC 回升、未知电池/不同配对、超出校准区间会报错。
这里假设实验相对 baseline 的变化可通过这一等效时间关系转移到参考电池，仍需实验验证，不保证消除全部电池/机体影响。

输入必需列：battery_id, drone_id, soc_start, soc_end, duration_s。额外列原样保留。
normalize_interval 可用于单个区间；normalize_csv 为每行使用同一个参考起始 SOC，适合同 SOC 横向比较。
不要把这些独立区间的 Bideal 降幅直接相加当作连续电池轨迹；连续轨迹必须将上一区间 q_end 作为下一区间 q_start。

```sh
python3 battery_normalization.py --model analysis_results/battery_normalization_candidate_20260908/model.json \
  --input YOUR_PROCESSED_INTERVALS.csv --output NEW_NORMALIZED_INTERVALS.csv --reference-soc 70
```

只写新输出文件，拒绝覆盖已有文件、原始输入，拒绝二次应用 bn_ 列。
示例 input/output 各五行，来自真实校准运行中 70% 附近约 25 秒的原始采样端点；actual duration 保留真实秒数，没有伪造精确 25 秒端点。示例只演示接口，不属于独立验证。

## 验证与限制

- 源文件 SHA256 在拟合前及输出完成后核对；sources.json 提供来源。
- 五次校准自身全段 H/Δt=1 是由端点拟合构造得到的，不是标准化效果验证。
- 历史检查固定本次全部参数，不在历史记录上重新拟合。旧实验风/高度条件未确认，跨日期老化等因素可能影响结果；仅作探索性稳定性检查。
- 目前每个当前电池/机体配对只有一次本批完整校准；在假定机体相同的前提下使用，不能单凭此拆分纯电池效应与机体效应。
- 25 秒内整数 SOC 变化只有几个百分点，量化误差不可忽略；不宜仅据单个窗口评价模型。

结论：本次历史检查中，高电量和中电量区间的差异没有缩小，仅低电量区间略有改善。因此只能说工具和候选参考已建立，不能说已验证能稳定消除电池差异。

| 历史公共 SOC 区间（只用于检查） | 原始率 CV | 换算因子 CV |
|---|---|---|
| high95_75 | 0.331 | 0.333 |
| low40_20 | 0.119 | 0.115 |
| medium75_40 | 0.049 | 0.057 |
| whole95_20 | 0.073 | 0.078 |

CV 是五块电池等权的总体标准差/均值；若同一电池有多次历史记录先取其均值。下降为差异缩小，上升为差异变大；不以历史结果调参。

## 文件

- model.json：机器可读取的五条个体曲线与一条 Bideal，以及拟合约束和来源。
- individual_segments.csv / individual_fit_quality.csv：15 段系数及原始数据拟合误差。
- mean_rate_segments.csv：相同 SOC 等权平均后的多段参考；含五块电池分别贡献的速率。
- reference_segments.csv / reference_search.csv：最终三段参考及完整分界搜索。
- reference_curve_derived.csv：构造曲线采样，仅作绘图/检查，不是真实观测。
- normalization_example_input.csv / normalization_example_output.csv：真实区间及保留原列的换算演示。
- historical_validation.csv / historical_summary.csv：冻结参数的历史检查。
- validation.json：数值与来源检查；six_curves.png / Bideal.png：静态图。

重新生成：运行 output_py/build_battery_normalization_candidate.py --output 一个不存在的新目录。不会改动已生成版本。
构建依赖 Python、NumPy、pandas、matplotlib；换算工具仅使用 Python 标准库。
离线测试：python3 -m unittest test_battery_normalization -v（不连接无人机）。
