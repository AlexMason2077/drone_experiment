# 第二步：50厘米风洞独立三阶段处理

独立检查全部138个候选CSV；目前21组经过筛选可用的记录、21个条件，其中Level1/2共20个，另外实际Level3单独保留。293条真实观测阶段率、315条曲线阶段系数，其中22条为缺阶段模型补全。这不是声明所有已采集实验都支持完整训练曲线；每份文件及剔除原因在audit。

## 方法

起飞、定位确认和非hover不参与拟合。每机去除最开始SOC一直不变的时间，从首次真实下降之后保留数据；后续整数SOC平台正常保留。有本机20%终点时保留至其首次20%，否则保留至实际hover结束，不因另一架先落地而提前截断本机。按每架实际电池自己的High/Medium/Low边界，分别用时间加权OLS拟合真实hover阶段率，恢复/重启块各有截距，不把缺失时间当飞行。

阶段标准化率 = 本机阶段实测率 / 固定九月份本机电池阶段基线率 × 固定参考电池阶段率。先有率，之后参考阶段时间 = 60 × 阶段SOC差 / 标准化率。参考边界仍是100/82/52/20，没有更改或重拟合baseline，没有按照起始电量不同而直接平均飞行时间。

缺少阶段时，曲线系数采用同配置同位置的风洞Medium相对因子（无Medium时用最近可用阶段）延伸；不加入实测率表。图上不放拓展标签，curve_stage_rates.csv的is_modeled和rate_origin、manifest保留来源。High支撑不足的情况会影响曲线的高电量阶段解释。原SOC实测对照图未加入；diamond_50_head_lv2_002数据保留。

## 文件

- all_wind_three_stage_curves.pdf：逐条件五架/位置的三段曲线；figures/含300dpi PNG及PDF/SVG。
- 50cm_wind_overview.png / .pdf：Level1/2全部条件总览；危险及没有完整可用记录的条件分别显示。
- observed_run_stage_rates.csv：仅真实观测阶段独立拟合，不含缺阶段补全。
- condition_position_stage_means.csv：风洞自己的同条件/位置/阶段均值。
- curve_stage_rates.csv / three_stage_curve_knots.csv：三段模型系数和四个节点。
- condition_coverage.csv：哪些阶段来自实测、哪些模型补全、哪些条件没有完整支撑。
- wind_run_audit.csv / wind_drone_window_audit.csv：文件筛选、各机初始平台删去秒数和各自hover窗口。
- manifest.json / validation.json：只读输入哈希和离线数值校验。

本步骤未读入250厘米实验结果，没有进行两类平均；未更新训练数据、模型、控制或数据库。五项危险配置全部排除。无完整可用Level1/2条件：front_50_head_lv2, vee_50_head_lv1, vee_50_tail_lv2, vee_50_side_lv2, echalon_50_head_lv1。Vee侧风有Level3实测，未偷偷改成Level2。

## 实测阶段支撑复核

stage_observation_support.csv记录每个阶段实际覆盖的SOC范围和初始平台删除秒数。wind_tunnel_front_50_tail_lv2_004的D1在38%结束；其低电量段只由实际可用区间拟合，延伸到20%的时间是模型计算。缺整段的22条补全，与有部分真实阶段但把阶段延伸到参考边界的情况分开记录。全部293条风洞原始阶段率已从各自原CSV重新拟合核对；最大误差小于1e-8 pp/min。
