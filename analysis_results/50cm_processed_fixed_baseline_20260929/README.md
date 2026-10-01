# 50厘米数据处理结果：固定九月份baseline

仅含50cm结果；风洞与旧250cm前进数据分开，没有跨协议平均。原始数据、九月份baseline、控制程序和训练集未改变。原SOC实测对照图未加入。

## 结果

- 风洞：检查138个CSV，保留21组、21个条件（Level1/2有20个；实际Level3另保留1个）。293条实测阶段率和315条曲线模型系数；22条缺整段用同位置风洞相对耗电因子完成，单独标记来源。
- 旧250cm前进：50cm间距范围内60组实验、281条有效Medium率、24个条件；19条无本机Medium前进时间保留audit。75cm未混入本包。
- 五项安全排除条件：column_50_head_lv2, column_50_side_lv2, column_50_tail_lv2, diamond_50_side_lv2, diamond_50_tail_lv2。整个条件从系数和曲线中排除，原失败记录仍在原数据库。
- 安全范围内但没有通过完整风洞记录筛选的条件：front_50_head_lv2, vee_50_head_lv1, vee_50_tail_lv2, vee_50_side_lv2, echalon_50_head_lv1。逐文件原因见wind_tunnel/wind_run_audit.csv，未用其他配置填造数据。

## 方法

旧前进率直接复用已有raw拟合，逐本机/电池/阶段乘“九月份目标基线率÷旧基线同阶段率”；然后换算到固定参考电池阶段率。没有把已标准化率再乘一次比例，实际前进时间不变。P5旧B15映射到当前B12的相对负载保持是一项换算假设。

风洞去除hover起初SOC不变的时间，后续整数SOC平台保留；按本机实际电池的阶段边界独立拟合。每机有20%终点则保留到自身20%，没有该终点则仅使用实际hover窗口。阶段率除本机固定基线率、乘参考率；最后按SOC差/阶段率计算时间。参考边界100/82/52/20保持原样，没有直接平均不同起始SOC飞机的飞行时间。

缺失阶段及部分阶段向参考边界的延伸可从is_modeled、rate_origin及stage_observation_support.csv追溯。图没有额外拓展标签。图中时间是参考模型计算值，不是原始飞行时长。wind_tunnel_front_50_tail_lv2_004的D1实际到38%，其低段延伸到20%是计算延伸。短阶段、少数SOC跳档和原前进零斜率继续保留质量标记。

## 查看和使用

- wind_tunnel/all_wind_three_stage_curves.pdf：风洞全部逐条件三段曲线。
- wind_tunnel/50cm_wind_overview.png / .pdf：Level1/2覆盖总览。
- 两套目录各自figures/：每个条件3300×2040 PNG及PDF/SVG矢量图。
- wind_tunnel/observed_run_stage_rates.csv：293条真实观测阶段率。
- wind_tunnel/curve_stage_rates.csv：315条完整曲线模型系数，含缺段补全。
- forward_250cm/adjusted_run_drone_rates.csv：281条已换算前进Medium率。
- forward_250cm/condition_position_medium_means.csv：各条件五位置均值；仅前进数据。
- source_stage_availability.csv：两来源观测支撑数量，未计算平均。共有94个同条件/位置/阶段有两类观测率，可用于以后第三步；旧前进只支持Medium。
- fixed_september_baseline.json：与原模型字节相同的只读副本。
- manifest.json / validation.json / summary.json：来源哈希、完整范围及离线校验。

当前为独立分析结果包，未自动加入训练标签。模型补全不算新增实测。后续若平均，应先分开计算各自重复实验的阶段均值，再在相同条件、位置、阶段上平均耗电率，之后算参考时间。
