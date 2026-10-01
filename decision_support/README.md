# Decision-Support Platform

面向单个五机 swarm 的独立桌面端原型。英文界面，默认 Demo，包含完整的任务观察、配置推荐、指令反馈与节点充电演示。运行时不导入项目原有的 `app.py`、数据采集器或无人机 SDK。

## 本地打开

在项目根目录运行：

```sh
python3 -m decision_support.server
```

浏览器访问 **http://127.0.0.1:8767**。停止本地服务使用 Ctrl+C。可通过 `PLATFORM_PORT` 修改端口。当前运行依赖 Flask，Leaflet 脚本随项目提供；街道底图在线加载 OpenStreetMap.de 瓦片，加载失败时使用本地 SVG。无需地图密钥。建议桌面宽度 1280–1920px；详细记录在下方展开。

## 已实现

- 五节点、四 segment 的预定义路线：Sydney Central Medical Centre → Rooftop A → Rooftop B → Rooftop C → Royal Prince Alfred Hospital。
- `/mission-input` 是单独的任务输入页：provider 可在地图上依次选起点、途经点和终点，并分别填写 D1–D5 的起始电量（0–100%，不预设满电）。完整输入可保存为当前浏览器内可继续编辑的草稿。该页面目前只负责输入与路线预览，尚未把草稿送入 Demo 的固定场景或 Live provider，因此保存不触发决策或飞行。
- 真实地理道路、绿地与建筑轮廓；点击节点显示充电位，点击 segment 显示上下文，支持缩放和拖动。
- 当前 segment / interval / interval 进度 / 下一次决策倒计时 / 独立的室内剩余距离。
- 首页显示当前配置，展开 Decision details 可查看推荐配置；Front、Column、Vee、Echelon、Diamond 的坐标读取项目已有 layout 数据，支持 0.50m 和 0.75m。
- Swarm status 显示风况、充电位与当前执行配置。已匹配到 Applied 决策时以 Best configuration 标出 Formation 和 Inter-drone spacing；待执行推荐仍在 Decision details 中单独显示。航段与飞行状态并入左上 Arrival 时间卡片，左侧面板高度参与布局，避免遮挡 Decision details。
- 论文截图可使用 `/?demo-interval=3` 和 `/?demo-interval=4`，分别显示固定的 Vee 与 Diamond 演示快照。两者不推进或修改常规 Demo 会话，风况改变、充电位保持 3/6；支持 2989 × 1720 的完整页面截图。
- 目标队形图按配置坐标绘制，所有编队在原卡片内放大位置间距并按视框边界调整比例，保留圆点和同步缩放的底部标尺；Column 在 P1–P2 之间另标中心距。实际位置遥测单独适配视图，不以目标间距标注实测距离。
- D1–D5 身份固定，队形图、状态卡、地图机群标签、候选分配与充电安排同步高亮。
- `Recommended → Sent → Acknowledged → Applied` 状态；执行反馈返回前保持原执行配置。待确认超过期限有提示。
- 总剩余时间为 `Arrival in + Charging after arrival`。后者包含等待，充电目标 99%。每次 Demo 决策还冻结改变前的执行配置与推荐配置在同一输入下的预测，供界面比较预计时间差。
- 首页只展示当前执行配置的队形图。`Ready at next node` 卡片右侧展示当前 segment、下一节点下最近一次已执行配置变化的预计时间差，并标注 decision、interval 与数据模式。未发生已执行变化、比较数据缺失或机群已到达节点时不显示该数字；当前 ETA 仍由当前预测提供，时间差保持决策时同输入下的冻结值。可见历史取决于 provider 返回的记录，Demo 快照最多保留最近 40 条。
- 历史决策、候选列表、预计充电分组与执行状态、事件记录可展开。浏览历史不改变当前任务。
- Demo 场景按钮：风况变化、充电位变化、位置交换、下一 interval、待确认、到达节点、充电完成、下一 segment、缺失电量、延迟输入和旧结果。
- Live 未连接状态与有标签的固定 Demo 录制回放。暂停自动决策与飞行进度分离。

## 三种数据模式

**Demo**：数据、候选顺序、耗电率和执行反馈均为明确标记的演示设置。后端按这些设置计算时间；候选并非真实研究神经网络的输出。`data/settings.json` 中的五个 candidate limits 仅属于该演示 provider，不是论文验证结果。

第二段 N1 → N2 的演示从 interval 3 开始，interval 1–2 已有脚本化历史记录。后续风况与可用充电位既有保持不变的阶段，也有变化的阶段。时间线显示每次已发生决策的输入和推荐配置；未到达的 interval 不显示为已决策。充电时间使用 `demo_charge_curve_minutes` 演示参数，当前脚本的总预计时间低于一小时；该参数不代表实测充电性能。

论文界面示例中，S2 interval 3 → 4 从低强度顶风转为低强度侧风，N2 的可用充电位始终为 3/6。风况变化后，已执行配置从 Vee 0.50 m 切换为 Diamond 0.75 m。Demo 为此采用脚本化的侧风耗电倍率：Vee 0.50 m 在侧风基本倍率上再乘 `1.747`，Diamond 0.75 m 的侧风倍率设为 `1.03`。在 interval 4 同一次演示输入下，保留 Vee 的预计 ready 时间为 3374 秒，改用 Diamond 为 3109 秒，故界面显示预计缩短 265 秒（4 分 25 秒）。这些倍率及收益仅为演示交互而设，不是实测或研究模型的效果数据。

**Replay**：回放程序生成的固定 Demo 录制帧，保留原时间戳，支持播放、暂停和定位。不能下发命令。它不是实验飞行记录的回放。

**Live**：通过 `PROVIDER_SNAPSHOT_URL` 读取单独提供的后端 JSON 快照。未配置服务或连接失败时显示空状态，不回填 Demo 电量。此版接口为只读；实际模型推理、控制下发与实机反馈仍需接入 provider 服务。界面不存在起飞、返航、急停等实机控制入口。

## 地图数据与节点

地图在线加载 OpenStreetMap.de 瓦片，覆盖整个页面并可拖动；本地 OpenStreetMap 地理数据生成的 SVG 用作加载回退。地图内显示数据与瓦片来源署名。用于本地 SVG 的道路、绿地与建筑数据于 2026-09-27 获取。许可与署名见 https://www.openstreetmap.org/copyright ，也可在 Data & provenance 对话框查看出处。

两个端点使用医疗机构公开地址的地图参考点：

- Sydney Central Medical Centre：Level 3 Suite 306, 451 Pitt Street, Haymarket。来源 https://internationalstudents.health.nsw.gov.au/healthcareservice/sydney-central-medical-centre/
- Royal Prince Alfred Hospital：50 Missenden Road, Camperdown。来源 https://www.nsw.gov.au/health-and-wellbeing/health-infrastructure-projects/royal-prince-alfred-hospital-redevelopment

三个中间节点位于 OSM 建筑轮廓内，分别为 way 548918549、369353658、205131017。它们是场景屋顶节点；屋顶适用性、充电设施、路线审批和实地坐标精度未经核验。建筑轮廓不构成实际可起降屋顶的证明。界面只标 Rooftop A/B/C，省去无关 POI。

`data/route.json` 保留各节点来源与建筑 footprint。扩大后的地理数据存为 `data/sydney_osm_expanded.xml.gz`，`tools/fetch_map.py` 可重新获取，`tools/build_map.py` 生成不带无关 POI 的本地 SVG。先前的 `data/sydney_osm.xml` 保留作为原始记录。

城市地图与室内实验分开。Demo 为 25 秒一个 interval、每 interval 2.5m 的配置。城市路线没有假定航程距离或缩尺比例；地图机群标记根据 Demo 进度示意。Live 模式仅在收到 GPS 来源位置时显示地图机群标记。室内剩余距离不作为城市飞行距离。

## 后端快照约定

所有日期使用带时区的 ISO 8601 字符串；持续时间为秒，距离为米。配置几何与内部电池归一化由 provider 负责。前端不训练模型、计算候选成本或将内部电量替代实际报告电量。

可参考 `/api/snapshot?mode=demo` 的 JSON 形状建立 Live adapter，但必须替换全部演示值与来源，并明确返回 `mode: "Live"`。

| 对象 | 主要字段 | 含义 |
|---|---|---|
| 顶层 | mode, source, provider, observed_at, connection | 数据模式、来源、服务快照时间和连接状态 |
| mission | id, name, status, segment_id/index, from_node, target_node | 当前任务与路线位置 |
| mission | interval, interval_count, interval_progress, next_decision_s, period_s | 由 provider 提供的 interval 进度与决策周期 |
| mission | remaining_indoor_m, remaining_map_m, mapping | 分开提供实验与地理距离；未知为 null |
| drones[] | id, battery, source, connection, quality, observed_at, position, charging_state | D1–D5 的实际报告电量/明确来源的回放或估计值；缺失为 null |
| conditions | wind, level, airflow, k, capacity, observed_at | 相对风况与气流方向；可用位数与节点总容量分开 |
| configurations[] | id, formation, spacing_m, positions[], spacing_reference, frame | 五位置的局部米制坐标、名义间距对应的位置对；+Y 为前进方向 |
| applied | config_id, assignment, revision, source | 最后收到执行反馈的配置，D1–D5 → P1–P5 的一一分配 |
| formation_positions（可选） | frame=local_forward, unit=m, source=Telemetry, observed_at, positions[{drone_id,x,y}] | 五机局部位置，+Y 朝前；完整且有来源时显示 Reported positions，否则显示 Target layout |
| geo_position（可选） | lat, lon, source=GPS, observed_at | 独立地理位置；Live 未提供 GPS 时不在城市图上推测机群位置 |
| decision | id, sequence, segment_id, interval_id, interval, input_as_of, input_snapshot_id | 决策身份、递增次序和使用的输入快照 |
| decision | model_version, rate_version, catalog_version, k, candidate_limit, candidates, recommended | 使用的模型与数据版本、该 K 对应的候选设置、返回列表和最小时间结果 |
| decision.comparison | before, after, saved_s, input_as_of, quality | 同一次决策输入下的前后配置及预计总时间差；缺输入或无候选时为 null |
| decision | status, command_id, sent_at, acknowledged_at, applied_at, ack_due_at, feedback[] | 指令生命周期；Applied 需有执行反馈 |
| forecast | arrival_s, charging_s, ready_s, as_of, basis, target_node, quality | 当前执行方案的后端预测；总时间等于前两项之和 |
| schedule[] | drone_id, group, wait_s, charge_s, start_s, end_s, status | 相对于到达时刻的预计分组和时间；不表示预约成功 |
| history[], events[] | 历史决策与带时间的事件 | 可展开查看，不覆盖当前执行上下文 |
| settings | candidate_limits_by_K, decision_period_s, charge_target_pct, battery_alert_rules | 必须来自已加载 provider/model 设置，不沿用 Demo 数值 |

`decision.comparison.before` 和 `after` 均包含 `config_id`、D1–D5 的 `assignment`、`arrival_s`、`charging_s` 与 `ready_s`。Demo provider 在决策时用同一电量、风况、可用充电位数与剩余飞行时间评估当时已执行的配置和推荐配置；历史比较结果不会随页面刷新或指令执行重算。`saved_s = before.ready_s - after.ready_s`，正值表示预计缩短时间，零值表示持平，负值表示推荐配置在该比较下更慢，不应改写成收益。该比较是演示估计，不代表实测节省时间。Live provider 若未提供同输入的比较结果，界面应显示不可用，不能用不同时间的当前预测和历史推荐结果相减。

正常 Demo 中没有实际位置遥测，队形视图标为 Target layout。Live 若要显示实际位置，应按前端支持的局部坐标约定传入单机位置；地理位置必须单独给 GPS 来源。正式接入时还需要 provider 的授权、输入检查、连接恢复和实机状态联调，本地 Demo 的通过不等于实机验证。

前端约每秒读取数据；Demo provider 使用独立时钟处理 25 秒决策边界和条件变更。页面刷新本身不触发新决策。旧请求响应与较旧 decision sequence 不覆盖较新的状态。暂停自动决策不会暂停 Demo 飞行进度。

## 验证

```sh
python3 -m unittest discover -s decision_support/tests -v
```

27 项自动测试覆盖：反馈到达前保持旧配置、待确认状态、位置分配、风况和充电位变化、周期决策、暂停语义、旧反馈处理、同输入配置比较、历史比较冻结、时间之和、充电分组、缺失与延迟数据、到达充电和下一 segment、五种队形坐标、模式隔离、固定演示状态与无硬件入口。

浏览器检查包括：桌面排版、节点详情、D3 双队形高亮、Live 空电量、候选列表和回放操作。验证过程不连接实机。
