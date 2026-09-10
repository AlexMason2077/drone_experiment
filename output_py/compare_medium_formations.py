"""Cross-formation comparisons using only the frozen real Medium profiles."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from output_py.interpret_medium_decisions import BASE, OUT


def name(formation):
    return "echelon" if formation == "echalon" else formation


def configuration(row, prefix):
    return f"{name(row[prefix + '_formation'])} {row[prefix + '_spacing_cm']} cm"


def geometry_contrast(a, b, wind):
    """Explain geometric differences without assigning unmeasured power components."""
    pair = frozenset((a, b))
    if pair == {"front", "column"}:
        if wind == "side":
            return ("The front row lies along the lateral wind component, allowing serial sheltering but also repeated wake encounters; "
                    "the column is transverse to that component, exposing its members more independently. Sheltering changes the thrust needed to resist drag, "
                    "while wake encounters change the rotor effort needed to produce that thrust.", ["R7", "R10"])
        return ("Front places neighbours side by side, whereas column puts them in a longitudinal chain. "
                "The chain provides sheltering but also repeated encounters with wind-convected rotor wakes; front avoids this serial exposure. "
                "A downstream rotor in adverse inflow needs more motor effort to restore thrust, which can offset the column's drag reduction.", ["R7", "R10"])
    if pair == {"column", "echalon"}:
        return ("Column repeats neighbours on one straight line; echelon adds lateral offset as well as longitudinal separation. "
                "This changes whether a downstream rotor samples the descending wake core or its edge, and therefore the rotor speed required to maintain thrust. "
                "Staggering also redistributes the moments that the attitude controller must correct.", ["R10", "R5"])
    if pair == {"front", "echalon"}:
        if wind == "side":
            return ("Under sidewind, the front row concentrates upstream sheltering along one line, whereas echelon offsets successive neighbours in both horizontal directions. "
                    "The offset changes wake interception and the distribution of drag-compensation and attitude-control demands among positions.", ["R7", "R5"])
        return ("Front keeps the drones at one longitudinal station, whereas echelon staggers successive drones both laterally and longitudinally. "
                "The transverse row avoids a direct follower chain; the diagonal layout changes wake-core versus wake-edge exposure. "
                "These two mechanisms need not produce the same total discharge or the same worst-position discharge.", ["R10", "R6"])
    contrasts = {
        frozenset(("front", "vee")): "Front uses one transverse row; vee divides the drones between two staggered arms. The arms introduce longitudinal follower relationships and lateral offsets that change sheltering and rotor-wake interception.",
        frozenset(("front", "diamond")): "Front keeps all positions at one longitudinal station; diamond introduces a centre and fore-aft positions as well as lateral neighbours. Diamond therefore combines streamwise sheltering with wake interactions from more than one direction.",
        frozenset(("column", "vee")): "Column puts all drones in one longitudinal chain; vee splits the follower relationships between two laterally separated arms. Splitting the chain changes both direct wake exposure and the benefit of upstream sheltering.",
        frozenset(("column", "diamond")): "Column repeats neighbours along a single longitudinal line; diamond spreads them across both horizontal axes while retaining centre-line interactions. This redistributes sheltering and wake-induced loading between central and lateral positions.",
        frozenset(("echalon", "vee")): "Echelon uses one staggered sequence; vee uses two staggered arms. The different neighbour offsets change which wake regions intersect each rotor, so the two layouts can distribute motor demand differently even at the same nearest-neighbour spacing.",
        frozenset(("echalon", "diamond")): "Echelon shifts successive neighbours laterally along a diagonal; diamond retains centre-line fore-aft interactions as well as lateral neighbours. The diagonal can move rotors away from direct wake cores, whereas the diamond combines this lateral offset with sheltered centre-line positions.",
        frozenset(("vee", "diamond")): "Vee has two staggered arms; diamond adds a central position and fore-aft alignment. The added centre-line interactions change the mixture of upstream sheltering and wake-induced loading, redistributing demand between the most exposed and most sheltered positions.",
    }
    body = contrasts[pair]
    if wind == "side":
        body += " With lateral wind, the streamwise order follows the crosswind component rather than the direction of travel."
    elif wind == "tail":
        body += " In tailwind, wake convection can reverse the upstream-downstream order relative to headwind."
    body += " The resulting inflow and force imbalance change rotor thrust production and the motor corrections needed to maintain the flight path."
    return body, ["R10", "R6", "R5"]


def comparison_query(mode):
    # Both queries execute against the same frozen candidate scores and profiles.
    fields = []
    for prefix, alias, p in (("selected", "s", "ps"), ("comparator", "c", "pc")):
        fields += [f"{alias}.formation AS {prefix}_formation", f"{alias}.inter_drone_spacing_cm AS {prefix}_spacing_cm",
                   f"{alias}.total_time_seconds AS {prefix}_total_time_s",
                   f"{alias}.charging_seconds_by_position AS {prefix}_charging_seconds",
                   f"{alias}.charging_pad_groups AS {prefix}_pad_groups",
                   f"{alias}.charging_pad_loads_seconds AS {prefix}_pad_loads_s"]
        fields += [f"{p}.position_{i}_discharge_rate_Bideal_pp_min AS {prefix}_p{i}_rate" for i in range(1, 6)]
        fields += ["(" + " + ".join(f"{p}.position_{i}_discharge_rate_Bideal_pp_min" for i in range(1, 6)) + f") * 25.0 / 60.0 AS {prefix}_swarm_drop_pp"]
    restriction = "c.formation_rank = 1" if mode == "best_available_spacing" else "c.inter_drone_spacing_cm = s.inter_drone_spacing_cm"
    return """WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY charging_pad_availability, wind_direction, wind_level, formation
    ORDER BY total_time_seconds, inter_drone_spacing_cm
  ) AS formation_rank FROM candidate_rankings
)
SELECT s.charging_pad_availability AS charging_pads, s.wind_direction, s.wind_level,
""" + ",\n".join(fields) + """
FROM ranked s JOIN ranked c
 ON s.charging_pad_availability = c.charging_pad_availability
 AND s.wind_direction = c.wind_direction AND s.wind_level = c.wind_level
 AND s.formation != c.formation
JOIN profiles ps ON ps.wind_direction = s.wind_direction AND ps.wind_level = s.wind_level
 AND ps.formation = s.formation AND ps.inter_drone_spacing_cm = s.inter_drone_spacing_cm
JOIN profiles pc ON pc.wind_direction = c.wind_direction AND pc.wind_level = c.wind_level
 AND pc.formation = c.formation AND pc.inter_drone_spacing_cm = c.inter_drone_spacing_cm
WHERE s.rank = 1 AND """ + restriction + """
ORDER BY s.wind_direction, s.wind_level, s.charging_pad_availability, c.total_time_seconds, c.formation;
"""


def critical_group(row, prefix):
    groups = json.loads(row[prefix + "_pad_groups"])
    loads = json.loads(row[prefix + "_pad_loads_s"])
    k = int(np.argmax(loads))
    return "+".join("P" + str(p).split("_")[-1] for p in groups[k]), float(loads[k])


def describe(row, citations):
    a, b = configuration(row, "selected"), configuration(row, "comparator")
    drop_a, drop_b = row["selected_swarm_drop_pp"], row["comparator_swarm_drop_pp"]
    gap = row["comparator_total_time_s"] - row["selected_total_time_s"]
    relation = "lower" if drop_a < drop_b else "higher"
    relative = abs(drop_a - drop_b) / drop_b * 100
    p_a, q_a = critical_group(row, "selected")
    p_b, q_b = critical_group(row, "comparator")
    head = (f"Against {b}, {a} reduces computed total time by {gap:.2f} s. "
            f"Its summed five-drone Bideal SOC drop over 25 s is {drop_a:.2f} rather than {drop_b:.2f} pp "
            f"({relative:.1f}% {relation}). ")
    if drop_a < drop_b:
        head += "The selected configuration therefore has lower aggregate normalized discharge in this comparison. "
    else:
        head += "This is a charging-workload advantage, not an aggregate-discharge saving. "
    head += (f"The last-finishing charging pad serves {p_a} ({q_a / 60:.2f} min), "
             f"compared with {p_b} ({q_b / 60:.2f} min) for {b}. ")
    k = row["charging_pads"]
    if k == 5:
        ma = max(row[f"selected_p{i}_rate"] for i in range(1, 6))
        mb = max(row[f"comparator_p{i}_rate"] for i in range(1, 6))
        head += f"The maximum position rate is {ma:.2f} versus {mb:.2f} Bideal pp/min; with five pads this maximum sets completion. "
    elif k == 1:
        head += "With one pad, the decisive quantity is the sum of all five charging times. "
    else:
        head += "With parallel charging, the position rates matter through the jobs sharing the critical pad, not only their sum or maximum. "
    physics, ref_ids = geometry_contrast(row["selected_formation"], row["comparator_formation"], row["wind_direction"])
    if row["selected_spacing_cm"] != row["comparator_spacing_cm"]:
        physics += " This best-configuration contrast changes spacing as well as formation; the matched-spacing companion separates the geometry comparison."
    return head + physics + " " + " ".join(citations[r] for r in ref_ids), ref_ids


def build_comparisons(citations):
    connection = sqlite3.connect(":memory:")
    pd.read_csv(BASE / "candidate_rankings.csv").to_sql("candidate_rankings", connection, index=False)
    pd.read_csv(BASE / "preprocessed_75pct_250cm.csv").to_sql("profiles", connection, index=False)
    results = {}
    for mode in ("best_available_spacing", "matched_spacing"):
        sql = comparison_query(mode)
        (OUT / f"cross_formation_{mode}.sql").write_text(sql)
        data = pd.read_sql_query(sql, connection)
        records = []
        for row in data.to_dict("records"):
            row["key"] = f"{row['charging_pads']}|{row['wind_direction']}|{row['wind_level']}"
            row["wind_strength"] = "low" if row["wind_level"] == 1 else "high"
            row["condition"] = f"{row['wind_strength'].title()} {row['wind_direction']}wind"
            row["comparison_mode"] = mode
            row["selected_configuration"] = configuration(row, "selected")
            row["comparator_configuration"] = configuration(row, "comparator")
            row["gap_s"] = row["comparator_total_time_s"] - row["selected_total_time_s"]
            row["selected_total_time_min"] = row["selected_total_time_s"] / 60
            row["comparator_total_time_min"] = row["comparator_total_time_s"] / 60
            row["aggregate_discharge_advantage"] = row["selected_swarm_drop_pp"] < row["comparator_swarm_drop_pp"]
            row["discharge_difference_pp"] = row["comparator_swarm_drop_pp"] - row["selected_swarm_drop_pp"]
            row["aggregate_drop_reduction_percent"] = row["discharge_difference_pp"] / row["comparator_swarm_drop_pp"] * 100
            row["selected_rate_profile"] = ", ".join(f"P{i}: {row[f'selected_p{i}_rate']:.3f}" for i in range(1, 6))
            row["comparator_rate_profile"] = ", ".join(f"P{i}: {row[f'comparator_p{i}_rate']:.3f}" for i in range(1, 6))
            row["selected_critical_pad"], row["selected_critical_pad_seconds"] = critical_group(row, "selected")
            row["comparator_critical_pad"], row["comparator_critical_pad_seconds"] = critical_group(row, "comparator")
            row["explanation_en"], row["reference_ids"] = describe(row, citations)
            assert row["gap_s"] >= -1e-8
            records.append(row)
        pd.DataFrame(records).to_csv(OUT / f"cross_formation_{mode}.csv", index=False)
        results[mode] = records
    connection.close()
    assert len(results["best_available_spacing"]) == 120
    by_key = {}
    for r in results["best_available_spacing"]:
        by_key.setdefault(r["key"], []).append(r)
    assert len(by_key) == 30 and all(len(v) == 4 for v in by_key.values())
    results["by_key"] = by_key
    # The appendix records all five rate vectors and both comparison designs.
    text = ["# Cross-formation comparison of the frozen Medium decisions", "",
            "Every decision is compared with the best available spacing of each of the other four formations, under the same wind and charging-pad count. "
            "A separate matched-spacing CSV compares formations at the selected spacing wherever real eligible data exist. "
            "Neither design substitutes missing or excluded profiles. Summed Bideal SOC drop is a normalized battery-consumption metric, not a measured energy value in joules. "
            "P1-P5 identify the recorded positions within each formation; equal position numbers need not have equal aerodynamic roles. "
            "The geometry discussion is a mechanistic interpretation of the measured profiles, not a decomposition of their motor power.", ""]
    for key, group in by_key.items():
        r = group[0]
        text += [f"## {r['condition']}, {r['charging_pads']} charging pad(s): {r['selected_configuration']}", "",
                 "Selected position rates (Bideal pp/min): " + r["selected_rate_profile"] + ".", "",
                 "| Other formation's best configuration | Rates P1, P2, P3, P4, P5 (Bideal pp/min) | Summed 25 s SOC drop (pp) | Total time (min) | Extra total time (s) |",
                 "|---|---|---:|---:|---:|"]
        for other in group:
            rates = ", ".join(f"{other[f'comparator_p{i}_rate']:.3f}" for i in range(1, 6))
            text.append(f"| {other['comparator_configuration']} | {rates} | {other['comparator_swarm_drop_pp']:.3f} | {other['comparator_total_time_min']:.3f} | {other['gap_s']:.3f} |")
        text += [""]
        for other in group:
            text += [f"**Compared with {other['comparator_configuration']}.** " + other["explanation_en"], ""]
    (OUT / "cross_formation_comparisons.md").write_text("\n".join(text) + "\n")
    stats = dict(decisions=30, best_other_formation_comparisons=120,
                 matched_spacing_comparisons=len(results["matched_spacing"]),
                 selected_has_lower_aggregate_discharge=sum(r["aggregate_discharge_advantage"] for r in results["best_available_spacing"]),
                 selected_wins_despite_higher_aggregate_discharge=sum(not r["aggregate_discharge_advantage"] for r in results["best_available_spacing"]))
    results["summary"] = stats
    (OUT / "cross_formation_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    return results


if __name__ == "__main__":
    refs = json.loads((OUT / "literature_sources.json").read_text())
    citations = {r["id"]: f"[{r['id']}]({r['url']})" for r in refs}
    print(json.dumps(build_comparisons(citations)["summary"], indent=2))
