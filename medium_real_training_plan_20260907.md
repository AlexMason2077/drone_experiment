# Medium 真实实验数据训练规划

日期：2026-09-07。状态：实施规划，尚未按本规划重新清洗、生成训练集或训练模型。

## 1. 本次确定的任务

沿用论文的 shared state–candidate scorer：每条样本包含一个当前场景和一个候选 configuration，网络输出一个成本残差；还原成本后排序，形成 candidate list。Online 对列表逐项精算并推荐其中耗时最短的配置。

- 当前放电建模范围：75%–40%，在区间内采用恒定耗电率。
- 当前任务长度：250 cm；指令速度：10 cm/s；模型任务时长：25 s = 25/60 min。
- configuration = formation + spacing + 五架无人机到五个位置的一一分配。
- 完整配置索引最多 5 × 2 × 120 = 1,200；实际候选要经过数据覆盖、既有安全限制及 SOC 适用范围筛选。
- 优化指标：所有无人机充到规定目标 SOC 的最短总完成时间，包含充电位不足时的等待；不是五机充电时间的简单平均或总和。
- 固定飞行时间下，最小化充电完成时间与最小化“飞行时间 + 充电完成时间”排序一致。
- 现有统一充电模型先作为明确的假设保留：相同充电位，充到 99%，0%→99%标定 90 min。输出是该假设下的最优时间，不是实测充电时间。
- 250 cm/25 s 是本阶段固定决策任务。论文中 30 s 决策间隔在单个25秒片段内不会触发第二次决策，不能用这一试验声称已验证飞行中多次动态重规划。

## 2. 数据现状与首轮入口

经现有表核对：

| 项目 | 数量/状态 |
|---|---|
| 已输出的逐 run × drone 耗电表 | 156 runs / 780 行，混有范围外记录 |
| 严格 medium 且已输出的子集 | 134 runs / 670 行，55 个条件组合 |
| 旧筛选表中满足正式候选和 medium 条件的记录 | 168 runs，包含上述134；另34条待重新处理和QC |
| 单个条件的重复次数（134子集） | 27个条件3次、25个条件2次、3个条件1次 |
| 134子集各机处理后前进时间 | 约15.973–37.723 s，并未统一 |
| 134子集检测前进距离 | 约180.929–241.631 cm，并非全部实测250 cm |

168的可复现入口筛选为：`formal_clean_trajectory_candidate == True`，`run_start_soc_max_pct <= 75`，`run_end_soc_min_pct >= 40`；以 `(experiment_directory, run_id)` 去重。134再限制为 `selection_status == selected`。

这里的检测前进距离是经过轨迹重建、终点锚定与删除非前进片段后累计的量，不能直接当作完整原始飞行距离来判定“没飞够”。134中102次原命令250cm、32次原命令300cm后截取前段；统一窗口审计必须回查原始时序和坐标重建依据。当前旧表范围来自2026-05-15至2026-07-02的前进实验，不包含最新风洞悬停实验。

34条不能自动变成新增合格训练样本：21条过去因为每条件最多选3条而未选，13条因存在其他非平坦SOC曲线而未选。先重新处理，报告平坦SOC、量化误差和样本质量，不按希望得到的配置排名选样。

55个medium条件缺少：column/50/head/lv2，column/50/side/lv1，column/50/side/lv2，diamond/50/side/lv2，diamond/50/tail/lv2。缺少数据与既有unsafe判定分别记录。

首轮从134条已输出run出发完成端到端验证，再审计另外34条。最终数量以新QC清单为准，不把134或168当作已经统一25秒、2.5m的就绪数量。

## 3. 原始数据处理与可追溯性

保留原始归档；新处理结果使用版本目录，不覆盖原始传感器记录。每个结果保留 source path、run_id、处理窗口和处理版本。

1. 核对 formation、风向、风级、间距、drone–battery–position 对应；注册表不一致逐条定位，以实际飞行证据和明确更正记录确认。
2. 去掉准备、起飞、降落后的片段，识别悬停和校准。缺失遥测、错误pad、突然SOC回升、断连和碰撞保留QC记录。失败run、合并run、合成/插值片段必须有来源标记；合成片段不进入真实训练输入。
3. 检查五机共同时间轴。针对连续前进试验，优先取同一连续25秒窗口，逐机核对沿前进方向位移；暂以225–275 cm作为规划中的“约2.5m”容差，输出敏感性结果，不修改真实位移使其过关。
4. 原有每机单独累计forward-only时钟，与共同25秒窗口是不同口径。无共同窗口的run不能称为同步25秒飞行；可列为辅助实测率来源，其到25秒的换算明确属于模型计算，不混入严格等时长主分析。
5. 记录原始电量下降、有效运动时间、实际位移、Bideal曲线和实测拟合率。以论文的曲线斜率法为主，端点下降/时间作为敏感性对照。平坦的整数SOC曲线不能解释为真实零耗电。
6. medium模型要求起始SOC和模型预测片段末SOC均在40%–75%内。不要把旧30%安全阈值误当成medium模型允许外推到30%的依据。
7. wind tunnel悬停单独标识协议，不能自动当成前进250cm数据。后续可做独立模型或验证两种协议的转换关系。

处理输出明确分为“共同窗口实测量”和“使用实测速率计算的25秒任务量”。首轮如果没有足够的共同窗口，报告数量和覆盖缺口，不补造测量值或默默放宽标准。

## 4. 电池与位置的换算

现有定义：`a_b = Bideal_hover_rate / battery_b_hover_rate`；`Bideal_drop = physical_drop × a_b`。

对于候选C把drone i分配到位置p(i)：

```
r_ideal_i = R[wind, level, formation, spacing, p(i)]
r_physical_i = r_ideal_i / a_battery_i
arrival_soc_i = soc_i - r_physical_i × (25/60)
```

SOC保留实际电池百分比口径。位置变化只改变位置率，电池身份和转换系数始终跟随无人机。Bideal减少电池差异，并不证明所有机体/桨叶效应都已消除；位置转用属于需验证的建模假设。

冻结一版经过medium适用性检查的battery calibration；缺失电池系数不默认填1。当前放电转换系数不能替代电池的充电曲线；首轮统一充电模型的限制需在结果中说明，后期独立充电曲线补充后同步重算标签。

存量baseline系数拟合范围为75%→30%或36%，不能称为已经严格按75%→40%拟合；要先做medium区间重拟合/敏感性检查。现有校准表仅含B10–B15，新电池（例如B06）需独立有效系数和适用范围证据。

已核对现有oracle、训练features和online可行性过滤都存在直接用Bideal rate扣raw SOC的路径。三处必须用同一个换算函数，否则训练与部署会不一致。

## 5. 训练表格式与真实数据边界

实际ML训练单位：`一个真实来源场景 × 一个可评价候选配置`，不是一个遥测点，也不是一行包含1200个网络输出。

建议输出四张数据表及一个manifest：

| 文件 | 粒度和关键字段 |
|---|---|
| `real_medium_run_drone.csv` | run/window × drone；source_run_id、protocol、battery_id、observed_SOC、position、actual_duration、actual_distance、raw_drop、Bideal_rate、QC |
| `real_medium_run_bundles.csv` | 每个同步五机来源一行；五机SOC/电池/转换系数、实验f/d、五个位置的实测率和来源 |
| `train_state_candidate.parquet` | 一个来源场景/K/候选一行；26维X、exact_cost、lower_bound、residual_label、配置和来源ID、eligible |
| `configurations.csv` | configuration_id → formation、spacing、position_d1…position_d5 |
| `dataset_manifest.json` | 距离/速度/单位、SOC边界、来源hash、过滤计数、feature顺序、充电/校准/标签版本、分组规则 |

训练样本构造采用以下约定：

- SOC取合格真实窗口的实际起始五机向量，电池取该run的实际映射；训练阶段不随机生成SOC、不补写SOC下降轨迹。
- 保留每个run的实测率五元组，首轮不只用一张池化平均率表替代所有真实记录。
- 该run首先支持其实际formation和spacing；对K=1…5及允许的位置排列构建监督任务。位置排列的成本是线性率/电池换算假设下的反事实计算，不是重新执行过120次飞行。
- 网络共享一个标量成本定义，所以可从不同真实run的f/d学习统一评分。上线和扩展验证时输入完整受支持配置集，跨formation和spacing排序效果必须单独检验。
- 在同一真实来源场景内做pairwise微调；不拿SOC不同的两个run直接比较标签并称为“同一个场景下两个候选的优劣”。同一场景若需跨f/d配对，要有明确版本的匹配实测率来源，并标明这是派生监督比较。
- K、候选枚举和精确成本属于派生监督信息。准确描述为“耗电依据全部来自处理后真实实验的训练集”；不是每一列都亲自实测。
- 每条cost保留来源。不能给未测试结构凭空设一个耗电率填满1200类。

按134个run计算，仅K×位置枚举的理论上限是134×5×120=80,400条state–candidate监督记录；若每场景保留前48和随机32则最多53,600条。所有数字均在QC、可行性、K=1/5特殊处理和去重之前。它们仍只依赖134次飞行，不是8万次独立实验。

## 6. 推荐输入vector：26维

| 索引（从1计） | 特征 | 数量 |
|---|---|---:|
| 1–3 | wind one-hot，顺序head/side/tail | 3 |
| 4 | wind_level | 1 |
| 5 | charging_pad_count/5 | 1 |
| 6–10 | soc_d1…soc_d5/100，固定无人机顺序 | 5 |
| 11–15 | formation one-hot，column/diamond/echelon/front/vee | 5 |
| 16 | spacing_cm/75 | 1 |
| 17–21 | 候选分配下D1…D5对应位置的Bideal实测拟合率，pp/min | 5 |
| 22–26 | 对应D1…D5的电池转换系数a_b | 5 |
| | 合计 | 26 |

这26维中的最后5维是电池系数，不是前面讨论过的sorted SOC。增加sorted SOC的对照版本为31维。暂不输入温度、姿态、固定距离、arrival SOC、charging jobs、job统计量和lower bound。

21维消融版本可把17–21改为已逆归一化的physical rate，并删除22–26；它已通过rate体现电池放电差异。不能同时使用Bideal rate、没有系数，又声称已处理可变电池。

全部特征顺序和标准化参数写入manifest。标准化只在训练集拟合；相同变换应用于验证和online。配置ID、位置排列ID、run ID保留为元数据，不再作为无序数字输入。

## 7. 标签与原残差训练模式

复用统一参考成本计算器：

```
Tf = 25/60 min
q_i = charging_model(arrival_soc_i)
Tc = exact_minimum_parallel_charging_makespan(q_1…q_5, K)
J = Tf + Tc
JLB = Tf + max(max(q), sum(q)/K)
y = log1p(max(J - JLB, 0))
J_hat = JLB + expm1(network(X))
```

网络直接输出一个非负log残差，不是1200维，也不是直接输出离散list。由模块排序后返回list。即使arrival SOC、q和LB不进入vector，它们仍在网络外用于标签、可行性和成本还原；这是保留原残差模式所必需的。

生成标签时先断言 `J >= JLB - tolerance`；仅对浮点容差内的负残差截零。超过容差的负值必须追查单位、电池换算或成本定义，不能由 `max(..., 0)` 静默掩盖。

K=1时Tc=sum(q)，K=5时Tc=max(q)，两者LB即精确值、残差为0。规划中这两类保留为解析精算基线和评测场景，部署可直接按LB选优，不把其结果归功于神经网络。主要NN训练与调度加速检验集中于K=2/3/4。

## 8. 训练、验证与独立检验

**训练**：采用以上真实来源run及派生K/排列标签，不读取旧随机25,000行训练表。首轮只用medium，候选预测到达SOC低于40%的不进入此模型可行域。

**扩展验证（遵循用户与老师当前安排）**：由冻结的medium discharge-rate模型生成场景，固定250cm与25秒，使用经过校准且训练范围覆盖的电池参数。SOC先限制在40%–75%，风况只取已覆盖条件。generated场景的起始SOC、25秒耗电、到达SOC和成本都保存生成版本与随机种子，source_type=`rate_generated`。

扩展场景拆成calibration和最终synthetic test，按base_state_id拆分；同一五机SOC/风况/电池组合的K副本全部同组。建议起点：500个base states用于选模型/TopN，另1,000个用于冻结后的评测，各含5种K。数量是拟定验证预算，不是当前已存在数据量；过滤无可行候选、去重并记录有效数量。

discharge-rate拟合若来自训练用真实run，此验证不是独立实测验证。它检验网络逼近该率/充电计算体系的能力。随机种子不同不消除共享实测率来源的依赖。

补充真实run分组诊断可暂时留出整次trial，不强行保证五折每个条件都有数据。所有同源窗口、五机记录、K副本、排列、断点续飞/合并组必须同组。若报告“未见真实run泛化”，率拟合/校准及其他数据驱动处理也应遵循来源拆分，不能先用全部run池化率再声称真实holdout独立。单run条件单列覆盖限制。

后续新增真实run作为冻结版本的外部检验，再决定是否纳入下一版训练。扩展验证能说明数值决策一致性，真实最优结论还依赖实测率和充电模型有效性。

## 9. 训练步骤与配置

1. 先独立验证标签：同电池不移动、只换电池、只换位置、K=1/5解析解、边界SOC、缺失校准、相同成本并列最优；所有模块得到同一arrival SOC和cost。
2. 小样本端到端：至少涵盖不同f/d与风况，输出可追溯的X、精确J、LB、y，确认候选还原正确再批量构建。
3. 首个可对照训练沿用原网络256→256→128、GELU、LayerNorm、Dropout0.08、单Softplus输出；原AdamW学习率7e-4、weight decay1e-4、Huber δ0.10、最多80epoch、patience10作为初始配置。网络输入宽度变为26。
4. 每场景保留至多前48个低成本候选和随机32个其他候选，不足则保留全部；重复数较多的run不能靠枚举行数获得不成比例权重。scaler和采样只使用训练来源。
5. 用calibration观察收敛和候选recall；保存最佳权重。旧batch2048只作起点，若有效样本不足则降低，记录实际每epoch更新次数。
6. 在收敛的回归模型上沿用原pairwise+BCE/Huber微调，学习率5e-5；同场景内比较，排除并列成本的错误有序标签。
7. 对照实验控制变量：26维原网络 vs 26维小网络128→64；21维已换算physical rate vs 26维Bideal+电池系数；sorted SOC只在明确改善独立calibration排序时保留。最终用同一未调参的synthetic test评估。
8. 用3个随机种子报告训练稳定性，不只选择最好一次。旧模型权重、旧scaler、旧TopN不能直接继承为新版本结论。

## 10. Candidate list与效果验收

按K和风向/风级报告：

- 最优集合Recall@N：shortlist是否包含任一精确最优候选；数值并列用统一容差判定。
- 近最优覆盖、所选配置regret：`Tc_selected - Tc_optimal`，报告均值、P95、最大值和适当的相对值。
- 候选数N与候选全集M；可行性失败/范围外状态单独计数，不从分母悄悄删除。
- 端到端latency，包括率查表、电池换算、可行性、LB构造、NN、排序和候选精算，报告中位/P95。与完整精算使用同一硬件和同一成本计算器。
- 与纯LB排序+同样TopN精算、完整精算和随机同预算筛选比较；明确K=1/5解析性能。
- 分别报告formation、spacing、位置分配的推荐情况，避免只在位置排列上有效，却声称学会了全部配置选择。

先沿论文alpha=0.90作为shortlist校准目标；评测N=1/3/5/10/20/50/100，选择calibration中各有足够样本的风况子组均达标的最小N(K)，冻结后测试。0.90意味着允许漏最优，不能宣传全局最优保证。实际使用可提高到0.95/0.99并报告预算代价，不能预先承诺结果。旧N={1:1,2:3,3:36,4:25,5:1}不沿用。

若要求严格保证全局最优：可利用未入选候选的有效LB与shortlist最佳精确成本比较；仍可能更好的候选继续精算直到可证明无改进。报告该扩展的真实计算次数，最坏仍可能回到完整精算。普通固定TopN只保证列表内最优。

## 11. 实施里程碑与交付物

| 顺序 | 工作 | 完成条件 |
|---|---|---|
| A | 数据来源和统一窗口审计 | 每run可追溯，主/辅助/排除计数，校准和protocol明确 |
| B | 冻结medium率与电池表 | 同一schema，所有可用位置、电池有依据；稀疏/缺失条件清单 |
| C | 统一电池换算和精确标签 | 训练/验证/online同函数，测试通过，交付一条真实来源样本及逐步标签计算 |
| D | 构建真实训练与扩展验证 | 表、manifest、source groups、无同源拆分泄漏，记录真实run数与派生行数 |
| E | 回归训练及排序微调 | 模型、scaler、训练曲线、随机种子和版本清楚 |
| F | 候选校准与最终比较 | Recall/regret/latency及基线齐全；TopN冻结 |
| G | 离线回放真实场景的online推荐 | 接收当前状态和电池映射，返回完整候选与推荐、可行域状态；再用新增真实实验检验 |

依赖顺序A→B→C→D→E→F→G。先交付A–D的可审查数据与计算，不在数据口径未确认时启动长时间训练。是否有明显在线收益由F决定；如果全枚举已经很快，必须如实报告，不人为增加baseline运行时间。

## 12. 代码落点与不兼容项

- 数据处理：`output_py/build_forward_discharge_rate_modeling_table.py` 作为读取/拟合逻辑来源；新版本显式medium筛选，单run保存。原每条件最多3条规则不自动继承为新训练集的必需限制。
- 参考成本：`ml_policy/oracle_optimizer.py` 和 `ml_policy/charging_model.py`；添加电池映射/scale、单位检查、模型适用边界。
- 训练：`ml_policy/train_controlled_k_candidate_cost_ranker.py` 和 `ml_policy/train_controlled_k_pairwise_ranker.py`；共享26维构造器，训练集入口改为真实run场景。
- 数据构造：现有cost matrix builder可复用精算方式，但不可继续默认读取旧随机场景表和旧全范围pool rate表。
- Online：`ml_policy/adaptive_topn_policy.py`；与训练共享features、校准、cost和可行性逻辑，加载本版TopN。
- 当前 `DEFAULT_RATE_TABLE_PATH` 指向 `configuration_condition_rate_bar_charts/pooled_configuration_drone_Bideal_forward_rates.csv`，不是当前medium过滤结果。实施时要显式版本化路径及schema adapter，不能混用旧表。
- 新的 `two_stage_formation_analysis.py` 属于另一分析路径，不作为本次把原候选残差网络换成另一架构的理由。

## 13. 本规划的证据与状态

主要已读证据：

- `analysis_outputs/forward_discharge_rate_modeling/selected_runs_by_database_cell.csv`
- `analysis_outputs/forward_discharge_rate_modeling/forward_discharge_rate_run_drone.csv`
- `analysis_outputs/forward_discharge_rate_modeling/battery_ideal_normalization.csv`
- `analysis_outputs/forward_discharge_rate_modeling/data_dictionary.md`
- 原候选ranker、pairwise、oracle、charging、adaptive_topn源码。
- 用户提供的 `IEEE_introduction_template (12).pdf` 第6–9页（本会话前一步已读取）。

核对结论：原有候选评分框架可继续；数据源、medium范围、电池逆归一化、标签构造和验证需一起更新。上述数量是现有文件审计结果；筛选后最终训练规模、模型精度、TopN和加速比仍待实施测定。本次仅交付规划文件，没有修改飞控、清洗归档或开始训练。
