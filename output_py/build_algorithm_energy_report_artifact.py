"""Build the canonical technical report for the cleaned swarm energy knowledge base."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "swarm_analysis" / "algorithm_energy_knowledge_base"


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()
    configs = pd.read_csv(OUT / "configuration_energy_knowledge_base.csv")
    lookup = pd.read_csv(OUT / "algorithm_configuration_lookup.csv")
    positions = pd.read_csv(OUT / "position_energy_extremes.csv")
    baselines = pd.read_csv(OUT / "single_drone_baseline_energy_models.csv")
    distances = pd.read_csv(OUT / "matched_distance_effects.csv")

    for df in (configs, lookup, positions):
        df["scenario"] = df.apply(
            lambda r: f"{r.wind_direction_short} · lv{int(r.wind_level)} · {int(r.distance)}cm", axis=1
        )

    winners = configs[configs.rank_within_wind_and_distance == 1].copy()
    winners["probability_best_pct"] = winners.probability_best * 100
    winners["energy_per_drone"] = winners.total_energy_median / 5
    winners = winners.sort_values(["wind_direction_short", "wind_level", "distance"])

    baselines = baselines.copy()
    baselines["battery_label"] = baselines.battery_id.astype(str)
    baseline_value = float(baselines.baseline_energy_median.mean())

    position_summary = []
    for formation, group in positions.groupby("formation"):
        low = int(group.lowest_energy_position.mode().iloc[0])
        high = int(group.highest_energy_position.mode().iloc[0])
        position_summary.append({
            "formation": formation,
            "typical_lowest_position": low,
            "lowest_position_frequency": int((group.lowest_energy_position == low).sum()),
            "typical_highest_position": high,
            "highest_position_frequency": int((group.highest_energy_position == high).sum()),
            "condition_count": int(len(group)),
            "median_position_spread": round(float(group.position_energy_spread.median()), 3),
        })
    position_summary = pd.DataFrame(position_summary).sort_values("formation")

    lookup_table = lookup[[
        "scenario", "formation", "rank_within_wind_and_distance", "run_count",
        "total_energy_median", "relative_vs_baseline_median_pct", "probability_best",
        "evidence_strength", "ranking_stability",
    ]].copy()
    lookup_table["probability_best_pct"] = lookup_table.probability_best * 100
    lookup_table = lookup_table.drop(columns="probability_best").sort_values(
        ["scenario", "rank_within_wind_and_distance"]
    ).round(3)

    sources = [
        {
            "id": "configuration-kb", "label": "Cleaned configuration energy knowledge base",
            "path": "swarm_analysis/algorithm_energy_knowledge_base/configuration_energy_knowledge_base.csv",
            "query": {"sql": "SELECT * FROM read_csv_auto('swarm_analysis/algorithm_energy_knowledge_base/configuration_energy_knowledge_base.csv')"},
        },
        {
            "id": "algorithm-lookup", "label": "Algorithm-ready configuration and position lookup",
            "path": "swarm_analysis/algorithm_energy_knowledge_base/algorithm_configuration_lookup.csv",
            "query": {"sql": "SELECT * FROM read_csv_auto('swarm_analysis/algorithm_energy_knowledge_base/algorithm_configuration_lookup.csv')"},
        },
        {
            "id": "baseline-models", "label": "Battery-specific single-drone forward baselines",
            "path": "swarm_analysis/algorithm_energy_knowledge_base/single_drone_baseline_energy_models.csv",
            "query": {"sql": "SELECT * FROM read_csv_auto('swarm_analysis/algorithm_energy_knowledge_base/single_drone_baseline_energy_models.csv')"},
        },
        {
            "id": "position-extremes", "label": "Within-formation position energy extremes",
            "path": "swarm_analysis/algorithm_energy_knowledge_base/position_energy_extremes.csv",
            "query": {"sql": "SELECT * FROM read_csv_auto('swarm_analysis/algorithm_energy_knowledge_base/position_energy_extremes.csv')"},
        },
        {
            "id": "distance-effects", "label": "Matched 75 cm versus 50 cm effects",
            "path": "swarm_analysis/algorithm_energy_knowledge_base/matched_distance_effects.csv",
            "query": {"sql": "SELECT * FROM read_csv_auto('swarm_analysis/algorithm_energy_knowledge_base/matched_distance_effects.csv')"},
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# 无人机编队飞行能耗知识库：清理、基线校正与条件排名"},
        {"id": "summary", "type": "markdown", "sourceId": "configuration-kb", "body": f"""## 技术摘要

本轮分析保留 **206 次完整五机实验（1,030 条无人机记录）**，形成 **57 个可用的 formation × wind direction × wind level × distance 条件单元**。结果表明不存在一个在所有风况和间距下都稳定最优的 formation：点估计中 front 获得 5 个条件第一，column 3 个，diamond 2 个，vee 与 echalon 各 1 个。

但是排名不确定性仍然很重要：57 个 formation 排名中只有 1 个达到 strong，10 个为 moderate，其余 46 个为 uncertain。因此，本报告中的排名适合作为后续算法的**经验先验与查找表**，不应暂时解释成确定性的空气动力学定律。"""},
        {"id": "metric", "type": "markdown", "sourceId": "baseline-models", "body": f"""## 核心指标与 baseline

直接比较电量百分比下降会受到初始电量和电池非线性放电的影响。本分析先用每块电池的 hover 曲线把电量变化转换为 **equivalent hover seconds**，再扣除静止等待，并把 300 cm 任务统一缩放至前 250 cm。五块 baseline 电池的 250 cm 单机前飞中位数平均为 **{baseline_value:.2f} equivalent-hover-seconds**；B12 按实验假设使用 B15 的曲线。

下图只标电池编号，不标无人机编号。"""},
        {"id": "baseline-chart", "type": "chart", "chartId": "baseline-energy"},
        {"id": "winner-finding", "type": "markdown", "sourceId": "configuration-kb", "body": """## 每种风况和间距下的最低能耗 formation

点估计最优 formation 会随 wind direction、wind level 与 inter-drone distance 改变。例如 head wind lv2 下，50 cm 的最低点估计是 vee，而 75 cm 是 diamond；这两个结论的 bootstrap 稳定性都较弱。column 的 50 cm、lv2 head/side 条件因碰撞风险主动停止，未填补或推断。"""},
        {"id": "winner-energy-chart", "type": "chart", "chartId": "winner-energy"},
        {"id": "winner-confidence", "type": "markdown", "sourceId": "configuration-kb", "body": """## 排名必须连同置信度使用

柱高表示该条件下 formation 在 bootstrap 重采样中成为最低能耗方案的比例。只有 side wind lv2、50 cm 下的 front 达到 strong；其余条件应保留候选集合，而不是只保存一个“冠军”。"""},
        {"id": "confidence-chart", "type": "chart", "chartId": "winner-confidence"},
        {"id": "position-finding", "type": "markdown", "sourceId": "position-extremes", "body": """## formation 内部的位置负担

每个条件均分别计算位置 1–5 的 baseline-relative 能耗。位置差异随 formation 和风况变化，并且电池身份与位置尚未完全解耦，因此这里报告的是经验位置负担，不作纯空气动力学因果解释。下方摘要显示每种 formation 中最常出现的最低/最高能耗位置及其出现次数；完整的条件级位置曲线在报告同目录的 charts 文件夹中。"""},
        {"id": "position-table", "type": "table", "tableId": "position-summary"},
        {"id": "scope-method", "type": "markdown", "body": """## 数据范围与方法

1. 只读取并处理 `db_copy_for_cleaning`，原始 `database` 保持不变。
2. 只比较纯前飞段；75 cm 布置中 300 cm 的实验按要求截取/归一到 250 cm。
3. 根据轨迹连续性修复 Mission Pad 1–8 重复使用造成的数米级坐标跳变。
4. 对 column 等错峰启动策略，将静止等待从能耗尺度中扣除；同时用 10 cm/s 的物理下限约束明显低估的移动时间。
5. 以每次实验为单位先汇总，再用条件单元的中位数和 IQR 比较，避免 front_50 重复次数较多以及不同起始 SOC 对普通 mean 的偏置。
6. 使用 4,000 次 bootstrap 检验各条件 formation 排名的稳定性。"""},
        {"id": "limitations", "type": "markdown", "body": """## 局限性与质量判断

- 数据足以建立**可追溯的初版经验知识库**，但尚不足以支持所有条件的确定性排序。
- 单元样本量为 1–8 次；1–2 次标记为 low，3–4 次为 moderate，5 次以上为 higher。
- 449/1,030 条记录需要移动时间物理下限校正，说明坐标/Mission Pad 遥测无法完整记录所有主动前飞时间。
- 电池百分比是整数遥测，39 条校正后无人机级估计出现零或负值；它们作为量化噪声标记保留，但没有任何条件级位置中位数为负。
- 两次实验的五机总量为非物理负值，已从 configuration 聚合中排除，同时保留在审计行级文件中。
- `diamond × 50 × side × lv2` 只有缺失电池/异常实验，因此没有可用排名。
- `column × 50 × head × lv2` 与 `column × 50 × side × lv2` 是安全性缺失，必须在论文中如实说明，不能作为随机缺失处理。"""},
        {"id": "lookup", "type": "markdown", "sourceId": "algorithm-lookup", "body": """## 可供后续算法读取的查找表

下面的精确查找表同时提供总能耗中位数、相对 baseline、位置 1–5 的能耗、样本量、证据等级以及 bootstrap 最优概率。算法阶段应读取这些数值及其不确定性，而不是只读取排名。"""},
        {"id": "lookup-table", "type": "table", "tableId": "configuration-lookup"},
        {"id": "next", "type": "markdown", "body": """## 下一步数据工作

1. 优先补测 low-evidence 单元，并重复 head lv2 与 side lv2 中排名接近的候选 formation。
2. 通过轮换电池与位置进一步区分 battery effect 和 formation-position effect。
3. 用更高频的电流/电压或飞控功率记录替代整数电池百分比，可显著降低短航段量化噪声。
4. 在进入算法建模前，将 lookup 中的中位数、IQR、样本量和最优概率一起定义为模型输入与不确定性边界。"""},
        {"id": "questions", "type": "markdown", "body": """## 后续需要回答的问题

- 当多个 formation 的 bootstrap 排名重叠时，算法应采用保守上界、期望值，还是风险约束？
- 是否需要补充 formation 切换过程的时间和能耗？
- charging pad 的充电速率是否随 SOC 非线性变化，以及多机同时充电时是否相互影响？
- 风况预测的误差应如何映射到 formation 选择的不确定性？"""},
    ]

    charts = [
        {
            "id": "baseline-energy", "title": "Battery-specific single-drone forward baseline",
            "subtitle": "250 cm, stationary overhead removed; lower is better",
            "type": "bar", "intent": "comparison", "dataset": "baseline", "sourceId": "baseline-models",
            "encodings": {
                "x": {"field": "battery_label", "type": "ordinal", "label": "Battery"},
                "y": {"field": "baseline_energy_median", "type": "quantitative", "label": "Equivalent hover seconds"},
                "tooltip": [{"field": "baseline_energy_median", "format": "number"}, {"field": "baseline_runs", "format": "number"}],
            },
            "layout": "full",
        },
        {
            "id": "winner-energy", "title": "Point-estimate winner by wind and distance",
            "subtitle": "Median total energy for five drones, normalized to 250 cm; lower is better",
            "type": "bar", "intent": "comparison", "dataset": "winners", "sourceId": "configuration-kb",
            "encodings": {
                "x": {"field": "scenario", "type": "nominal", "label": "Wind · level · spacing"},
                "y": {"field": "total_energy_median", "type": "quantitative", "label": "Five-drone equivalent hover seconds"},
                "color": {"field": "formation", "type": "nominal", "label": "Formation"},
                "tooltip": [{"field": "formation"}, {"field": "run_count", "format": "number"}, {"field": "total_energy_median", "format": "number"}],
            },
            "settings": {"categoryLabelPolicy": "rotate"}, "legend": {"position": "bottom"}, "layout": "full",
        },
        {
            "id": "winner-confidence", "title": "How often the point-estimate winner remains best",
            "subtitle": "4,000 bootstrap resamples; higher means a more stable winner",
            "type": "bar", "intent": "comparison", "dataset": "winners", "sourceId": "configuration-kb",
            "encodings": {
                "x": {"field": "scenario", "type": "nominal", "label": "Wind · level · spacing"},
                "y": {"field": "probability_best_pct", "type": "quantitative", "label": "Probability best", "unit": "%"},
                "color": {"field": "formation", "type": "nominal", "label": "Formation"},
                "tooltip": [{"field": "formation"}, {"field": "probability_best_pct", "format": "number"}, {"field": "ranking_stability"}],
            },
            "referenceLines": [{"axis": "y", "value": 75, "label": "Strong threshold", "lineStyle": "dashed", "color": "neutral"}],
            "settings": {"categoryLabelPolicy": "rotate"}, "legend": {"position": "bottom"}, "layout": "full",
        },
    ]

    tables = [
        {
            "id": "position-summary", "title": "Most frequent low/high energy position by formation",
            "dataset": "position_summary", "sourceId": "position-extremes", "layout": "full", "density": "spacious",
            "columns": [
                {"field": "formation", "label": "Formation", "type": "text"},
                {"field": "typical_lowest_position", "label": "Often lowest", "format": "number"},
                {"field": "lowest_position_frequency", "label": "Occurrences", "format": "number"},
                {"field": "typical_highest_position", "label": "Often highest", "format": "number"},
                {"field": "highest_position_frequency", "label": "Occurrences", "format": "number"},
                {"field": "condition_count", "label": "Conditions", "format": "number"},
                {"field": "median_position_spread", "label": "Median spread", "format": "number"},
            ],
        },
        {
            "id": "configuration-lookup", "title": "Condition-level energy and ranking lookup",
            "subtitle": "Exact robust summaries; use evidence and stability columns with the point estimate",
            "dataset": "lookup", "sourceId": "algorithm-lookup", "layout": "full", "density": "compact",
            "defaultSort": {"field": "scenario", "direction": "asc"},
            "columns": [
                {"field": "scenario", "label": "Condition", "type": "text"},
                {"field": "formation", "label": "Formation", "type": "text"},
                {"field": "rank_within_wind_and_distance", "label": "Rank", "format": "number"},
                {"field": "run_count", "label": "Runs", "format": "number"},
                {"field": "total_energy_median", "label": "Total energy", "format": "number"},
                {"field": "relative_vs_baseline_median_pct", "label": "Vs baseline (%)", "format": "number"},
                {"field": "probability_best_pct", "label": "P(best) (%)", "format": "number"},
                {"field": "evidence_strength", "label": "Evidence", "type": "text"},
                {"field": "ranking_stability", "label": "Stability", "type": "text"},
            ],
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": "无人机编队飞行能耗知识库",
            "description": "Formation、风向、风速、间距与位置的 baseline-relative 能耗分析。",
            "generatedAt": generated, "blocks": blocks, "charts": charts, "tables": tables, "sources": sources,
        },
        "snapshot": {
            "version": 1, "generatedAt": generated, "status": "ready",
            "datasets": {
                "baseline": records(baselines[["battery_label", "baseline_energy_median", "baseline_runs"]].round(3)),
                "winners": records(winners[["scenario", "formation", "total_energy_median", "energy_per_drone", "run_count", "probability_best_pct", "ranking_stability"]].round(3)),
                "position_summary": records(position_summary),
                "lookup": records(lookup_table),
            },
        },
        "sources": sources,
    }
    path = OUT / "artifact.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
