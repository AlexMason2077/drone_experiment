# 50 cm 风洞三阶段曲线图集

25个纳入条件均有编号采集记录和五位置曲线。30种组合中按当前规则排除5种危险条件；Front、Vee、Echelon各6种，Column 3种，Diamond 4种。

## 文件

- `wind_tunnel/all_50cm_curves.pdf`：25页完整图册。
- `wind_tunnel/50cm_overview.png/pdf`：30格总览，5个排除条件显示灰色；所有小图时间尺度一致。
- `wind_tunnel/figures/`：25张独立PNG（3300×2040，300dpi）、PDF、SVG。
- `wind_tunnel/curve_stage_rates.csv`：375个阶段系数，组成125条位置曲线；系数与最新审查版本逐字节一致。
- `wind_tunnel/condition_coverage.csv`：编号采集覆盖、真实拟合支撑及模型阶段计数。
- `wind_tunnel/collection_inventory.csv`：原始候选文件清单；包括失败、prepare和空记录，不等同于有效训练实验数。
- `wind_tunnel/observed_run_stage_rates.csv`、`stage_observation_support.csv`：真实记录拟合及阶段支撑清单，缺段不填假观测。
- `provenance/`、`manifest.json`、`validation.json`：来源、拟合方法、模型先验及离线核对记录。
- `forward_250cm/`：先前按新基准调整的250cm结果原样保留，24条件、60次记录、281个实验/无人机系数；没有与风洞平均。

## 计数和解释

风洞纳入条件共有83个非空编号源文件，属于60个实验编号目录；其中3个源文件在注册表标为异常，采集覆盖计数不代表其全部可训练。338个真实记录阶段拟合支持334个条件/位置/阶段。最终375个曲线系数中308个直接来自阶段拟合，67个包含模型估计：41个没有本阶段实际拟合支撑，26个将实际拟合与另一档风速参考组合。所有相关来源及权重保留。

曲线按固定九月份参考电池的100/82/52/20边界绘制。时间由各阶段耗电率计算，不是不同初始电量的原始实验时长平均。每架无人机开头不变SOC时段已按此前方法排除。图保持中性条件标题，不显示repair或拓展标签。模型补全不作为新的实测记录；本次没有更新训练数据、原始记录、基准或飞行控制。
