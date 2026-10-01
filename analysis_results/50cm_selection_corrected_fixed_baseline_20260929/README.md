# 50厘米筛选修正结果

**安全范围的25个Level1/2配置全部有非空编号记录。之前把筛选未纳入写成缺少采集，是表述与筛选错误。**

本包保留旧250厘米换算结果，固定九月份baseline字节未变。两种协议未平均，训练集、控制器及原始实验未改。

## 这次修正

恢复Front head Level2、Vee head Level1、Vee tail Level2三个长记录配置，并将Echelon head Level1现有短记录逐机、逐阶段检查。原来一架出现异常标签就删除整个实验，现在逐无人机提取实际窗口；合并Front只用observed来源块，3095个模拟间隙行完全不拟合。

Vee head Level1和Echelon head Level1部分control_fault为持续状态标签，之后仍有实际空中耗电观测。仅纳入识别到已知Mission Pad且pad相对高度与ToF同时为60–100厘米的该类记录，保留fault_phase_quality_flag及different_assigned_pad_sample_count。不同Pad切换分开拟合；发生故障或离开分配Pad的实测率需定位控制质量复核，不自动认定为稳定编队训练标签。这个离线筛选不等于证明编队稳定、间距精确或实机安全。没有把异常落地、起飞、移动或定位修正段当作hover。

每个真实块单独截距；不把恢复或模拟间隔计入拟合时间。开头SOC不变的段去掉，后面整数SOC平台保留；按各自电池阶段先拟合率，再标准化，最后算参考100/82/52/20时间。缺整段只补在曲线模型系数表，实测阶段表不添观测。

本包有25个Level1/2五机三阶段结果。新增48条真实阶段率，原来293条及21个条件的数值逐项保持。

## 用户确认与短记录处理

- Echelon 50 head Level1：按用户确认允许使用各阶段的短记录，不要求同次飞行耗尽。实测拟合来自有实际SOC下降的可用窗口，缺整段只在模型系数中按同位置相对负载完成。004在本地Git首次提交也是约6秒的起飞记录，未找到完整历史副本；其中D3有47→44%下降，但阶段为takeoff且切到别机Pad，未冒充80厘米定位悬停Low观测。
- Vee 50 side：用户已确认Level3 003的实际风速为Level2，派生数据修正为Level2。CSV原文件未改，原Level3、实验ID、run_id及用户确认保留在wind_level_metadata_corrections.json。

## 文件

- wind_tunnel/all_wind_three_stage_curves.pdf、figures/：更新的每条件三段折线；PNG 3300×2040以及PDF/SVG。
- wind_tunnel/collection_inventory.csv、condition_coverage.csv：采集存在与分析归属分开列。
- wind_tunnel/observed_run_stage_rates.csv：真实阶段拟合。recovered_observed_stage_rates.csv只列新增记录。
- wind_tunnel/recovered_selected_real_samples.csv：新增拟合所用的每个真实样本与实际源时钟。
- wind_tunnel/curve_stage_rates.csv：曲线系数，缺整段来源在is_modeled和completion_anchor_stage。
- wind_tunnel/stage_observation_support.csv：实测SOC覆盖及初始平台删除秒数。
- forward_250cm/：上次已审核的281条50cm前进Medium率与24条件，逐文件复制未变。
- summary.json、validation.json、manifest.json：范围、复算、只读输入哈希及用户明确确认的标签修正。

旧包保留便于追溯；本包替代它的采集覆盖和筛选结论。25条件都有分析曲线，不代表每个阶段都有完整SOC实测，也未写入训练标签。特别是Echelon head Level1的缺阶段完成不应混入纯实测训练表。

## 独立复核

全部341条实测阶段率已经从各自原CSV重新选择并独立最小二乘复算，原始率最大误差1.8e-14 pp/min。source_stage_availability.csv列出两个来源同条件同位置同阶段的观测支撑；共有115个有双来源，未平均。echelon_004_local_history_check.json记录004在首次Git保存时已为6.062秒的起飞记录，时钟与SOC和当前相同。25张PNG均核验为3300×2040并检查图形排版。
