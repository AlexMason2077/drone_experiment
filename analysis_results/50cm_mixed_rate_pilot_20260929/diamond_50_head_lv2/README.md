# 单条件试处理：Diamond / 50 cm / Headwind / Level2

仅交付此例，等待用户看图确认后再继续批量。没有更新训练数据、激活模型、改动实机代码或原始记录。实测SOC对照图不进入结果集，原始 `_002` 实测数据保留。

## 平均方式

风洞各物理电池按自身High/Medium/Low阶段独立拟合，去掉各机hover开头不掉电的平台，保留各机至自身20%的时间。已算好的250cm v3耗电率直接复用，不重拟合。两类已在相同参考电池上，每类先算重复实验均值，再各占50%。

**现有250cm系数仅覆盖Medium**，因此本例只在中电量阶段做两类实测平均。High和Low有风洞实测的使用风洞率；P3/P4没有High观测，曲线中按该位置平均后Medium的相对耗电因子完成High。这两行明确记录为模型补全，不进入实测拟合表。

以P1为例，中电量风洞8.703709 pp/min，已有250cm均值5.934877 pp/min，平均7.319293 pp/min。统一参考Medium区间82→52，所以阶段时间=60×30/7.319293秒。先平均率，再计算时间；不对两个实验的原始时长做平均。

图中的横轴是统一参考电池从100%到20%的计算时间，和起始电量不同的原始实验时长含义不同。每个位置的三阶段率、四个折点以及复核信息均保留。PNG为3300×2040，PDF/SVG为矢量。

这是风洞悬停与250cm前进两种协议的描述性平均。旧参考电池模型保持candidate状态，本次未独立验证其物理有效性。P2/P5的风洞High各只有两档SOC，保留原试处理斜率并标记支撑较弱。其他飞机先落地后的本机hover仍按用户要求保留，因此不保证后段一直有五架同时悬停。

## 文件

- averaged_three_stage.png / pdf / svg：这一个条件的五位置标准化三段曲线。
- source_stage_means.csv：分别列出风洞/已有250cm均值与数量。
- averaged_observed_stage_rates.csv：可观测阶段平均；High/Low单来源未伪装成两类平均。
- curve_stage_rates.csv：图使用的15个阶段率及来源，P3/P4 High为模型完成。
- curve_knots.csv：100/82/52/20四个折点。
- measured_stage_rates.csv：只含真实观测拟合或已有处理系数。
- wind_window_audit.csv / reused_forward_records_audit.csv：窗口与前进复用依据。
- manifest.json / validation.json：来源校验、平均公式及单条件范围复核。
