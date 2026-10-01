# 两个独立处理步骤：固定九月份baseline

目前完成的是两套独立候选结果；未进行250厘米与风洞的平均，也未写入训练数据或在线模型。

1. step1_forward_250cm/：133组原有250厘米实验，625条Medium率、54条件（50cm 24条件 / 75cm 30条件）。按同无人机旧电池、同SOC阶段的基线，换算到九月份目标及固定参考尺度；实际前进时间不变。40条无对应阶段前进数据保留audit，当前危险配置排除。
2. step2_wind_50cm/：138个候选文件逐一筛选，21组可用、21条件，293条实测阶段率及22条缺整段模型补全。初始不变电量时间去除；各自独立拟合阶段率，然后计算参考100/82/52/20的时间。
3. 后续再平均：本次未执行。将来只有相同条件/位置/阶段且两套都有观测率时，才有两类实测平均依据。目前旧250厘米结果只支持Medium。

## 先看这个例子

- step1_forward_250cm/figures/diamond_50_head_lv2_forward_medium.png：历史前进数据换算后的Medium参考曲线。
- step2_wind_50cm/figures/diamond_50_head_lv2_wind_three_stage.png：同条件风洞三阶段曲线。
- 两套目录分别有逐条件PDF、系数CSV、来源audit与数值校验；曲线时间由耗电率算出，不直接平均实验实际飞行时间。

缺阶段和部分阶段延伸保留在来源表；图没有拓展文字标签。特别是P5旧B15映射到当前B12、日期间负载比例稳定，属于换算假设，不是已经验证的电池老化原因。没有改固定model、实测原件、控制器、训练集，也没有Git push。

复现先运行output_py/process_separate_forward_wind_latest_baseline.py的forward、wind两个独立命令（使用新的输出目录保留既有产物）；再参考output_py/audit_separate_baseline_outputs.py复核原始率和实际SOC支撑。
