# 单组 wind tunnel 三阶段试处理

实验：`wind_tunnel_diamond_50_head_lv2_002`，run `20260928_232343`。

使用已有 Bideal：High 100–82%、Medium 82–52%、Low 52–20%。

先去掉起飞、定位与开头不掉电平台，从有效悬停后的首个掉电样本开始计时；后续整数 SOC 平台保留。仅使用五机全部就位后、首架开始降落前的五机悬停区间。

按照原有等效悬停时间积分进行连续标准化，参考起点与首个实测掉电后的 SOC 数值对齐。先逐参考阶段初拟合，再将 82/52 连接连续性与首个掉电 SOC 锚点纳入联合最小二乘拟合。原始数据未修改。

P3–P5 完全没有本次 High 观测，试图采用同位置 Medium 相对 Bideal 耗电因子延长到 High。此为待确认的建模假设，全部来源与假设见 manifest.json，补出的点不参与拟合。P1 Low 和 P2 High 的有效覆盖很短。该试处理不是已激活的训练数据。

图中没有外推文字标签，原始范围、补齐范围及状态保留于数据文件。normalized_three_stage 为三段模型图，observations_and_fit 为有效观测与模型的对照。模型的 100→20% 总时长不等于这次实际飞行总时长；静态风洞悬停率不冒充 2.5 m 前进率。

stage_rates.csv：三个阶段的速率及来源；initial_plateau_and_window_audit.csv：每机剔除的平台时间及有效窗口；normalized_observed_samples.csv：由实测得到的连续参考 SOC；three_stage_knots.csv：三段连接点。
