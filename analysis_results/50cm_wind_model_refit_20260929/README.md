# 四组50 cm风洞曲线重新拟合

已重拟合用户指出的四个条件和用于参考的四个另一档风速条件。其余17个条件保留上一版系数，没有宣称本次全部重新验证。原始实测CSV、固定九月份baseline、旧版输出、实机控制代码及训练数据保持不变；250 cm结果原样保留，没有平均。

## 处理方式

1. 找回所有可用编号实验的实际悬停片段，不再要求整次实验必须飞到20%。使用名义hover、识别到指定Pad、Pad高度和ToF均60–100 cm的实际样本；故障阶段及失去/换Pad的样本不作为该配置的有效拟合点。这是离线数据筛选，不是实机安全验证。
2. 每架飞机去除开头不变的SOC时间。不同起飞、恢复、失去定位或源文件各有自己的时间块和截距，不跨停止时段拼接时间。
3. 按各自电池High/Medium/Low范围拟合SOC首次报告下降的时间点。中途整数平台仍计入相邻下降事件之间的实际时间。前一阶段在首次到达下边界时结束，下边界平台不再重复进入前一阶段。单个块至少三个SOC层级，累计实际观察下降至少3个百分点；只有一次跳变的4秒窗口不支撑整阶段率。
4. 实测候选耗电率仍按固定九月份模型逐阶段归一化。不同起始SOC只影响覆盖范围，不直接平均飞到20%的时间。
5. 修复图采用带参考的模型拟合：实际阶段覆盖越完整、不同编号阶段率越一致，保留实测拟合的权重越大。部分观测和异常阶段更多参考同队形、同风向、同位置另一档风速。权重及实测率、参考率、最终率都写入系数表。拟合权重是明确的建模规则，不是经验置信概率。
6. Level1缺段按用户授权的较弱耗电趋势补全。可靠配对Medium支持时使用实际比例；支持不足或数据不支持该趋势时，模型设定Level1率为Level2的95%，相应时间增加约5.3%。并不宣称所有实测Level1必然比Level2耗电慢。
7. 另一档风速的High也缺少实测时，使用固定参考三段形状和该位置实测中/低电量负载形成模型先验，单独标注来源；不将短High因子复制到整个中低电量阶段。未增加模拟行到原始或实测阶段表。

## 使用的两档比例

- echalon_50_head: Level2/Level1 耗电率比例 1.0526；来源 user_authorized_5_percent_L1_rate_reduction_scenario。
- front_50_side: Level2/Level1 耗电率比例 1.1433；来源 same_family_reliable_Medium_pairs。
- front_50_tail: Level2/Level1 耗电率比例 1.0526；来源 user_authorized_5_percent_L1_rate_reduction_scenario。
- vee_50_tail: Level2/Level1 耗电率比例 1.1332；来源 same_family_reliable_Medium_pairs。

## 模型时间变化

以下均为固定参考SOC 100%→20%的模型时长，不是直接测得的完整飞行时间。

| Condition | Position | Previous model (s) | Refit model (s) | Change (s) |
| --- | --- | --- | --- | --- |
| echalon_50_head_lv1 | 1 | 182.3 | 380.6 | 198.3 |
| echalon_50_head_lv1 | 2 | 165.9 | 399.2 | 233.3 |
| echalon_50_head_lv1 | 3 | 193.2 | 379.2 | 186.0 |
| echalon_50_head_lv1 | 4 | 453.9 | 406.1 | -47.8 |
| echalon_50_head_lv1 | 5 | 163.5 | 432.6 | 269.1 |
| front_50_side_lv1 | 1 | 459.5 | 471.3 | 11.8 |
| front_50_side_lv1 | 2 | 413.5 | 429.1 | 15.6 |
| front_50_side_lv1 | 3 | 777.8 | 394.3 | -383.5 |
| front_50_side_lv1 | 4 | 415.2 | 442.0 | 26.8 |
| front_50_side_lv1 | 5 | 433.1 | 423.1 | -10.0 |
| front_50_tail_lv2 | 1 | 255.9 | 401.1 | 145.2 |
| front_50_tail_lv2 | 2 | 254.4 | 417.2 | 162.8 |
| front_50_tail_lv2 | 3 | 210.2 | 394.6 | 184.4 |
| front_50_tail_lv2 | 4 | 323.5 | 384.3 | 60.7 |
| front_50_tail_lv2 | 5 | 206.9 | 387.6 | 180.8 |
| vee_50_tail_lv2 | 1 | 875.1 | 405.2 | -469.9 |
| vee_50_tail_lv2 | 2 | 404.9 | 392.5 | -12.4 |
| vee_50_tail_lv2 | 3 | 457.6 | 388.3 | -69.3 |
| vee_50_tail_lv2 | 4 | 454.1 | 412.8 | -41.3 |
| vee_50_tail_lv2 | 5 | 364.4 | 348.3 | -16.1 |

## 输出

- `wind_tunnel/repaired_four_overview.png/pdf`：四张修复图，保持原有五位置和参考曲线布局。
- `wind_tunnel/repaired_four_conditions.pdf`：四张单独图。
- `wind_tunnel/figures/`：25个条件PNG/PDF/SVG；本次改动8个条件。
- `wind_tunnel/curve_stage_rates.csv`：完整375阶段系数；`estimate_kind`区分实测拟合、带参考的模型估计及未复核旧候选。
- `wind_tunnel/refitted_real_run_stage_rates.csv`：从实际合法窗口重拟合的逐实验阶段率；没有参考风速填补的假观测。
- `wind_tunnel/repaired_eight_condition_stage_estimates.csv`：最终模型率、实测权重、参考条件、原始来源及缺段先验。
- `wind_tunnel/actual_SOC_crossing_samples.csv`、`actual_window_audit.csv`：实际拟合点与筛选说明。
- `wind_tunnel/cross_speed_factors.json`：配对记录比例和5%模型设定。
- `wind_tunnel/before_after_modeled_times.csv`：四组前后模型时长。
- `manifest.json`、`validation.json`：来源校验、方法、约束及离线检查。原始失败记录和所有危险条件排除规则保留。

这些结果是用户授权的修复模型版本。借用风速及缺阶段的估计保留出处，可用于审查模型曲线；在后续训练或论文实测性能结论使用时，应区分这些模型估计与原始实测证据。
