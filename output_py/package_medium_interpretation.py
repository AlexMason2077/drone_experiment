"""Package the audited decisions and reviewed literature as one canonical report."""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from output_py.interpret_medium_decisions import BASE, OUT, ROOT, digest
from output_py.medium_decision_mechanisms import interpretation, mechanism_sections
from output_py.compare_medium_formations import build_comparisons, geometry_contrast


def main():
    audit = json.loads((OUT / "numeric_audit.json").read_text())
    summary = json.loads((OUT / "audit_summary.json").read_text())
    refs = json.loads((OUT / "literature_sources.json").read_text())
    citations = {r["id"]: f"[{r['id']}]({r['url']})" for r in refs}
    generated = datetime.now(timezone.utc).isoformat()
    if (OUT / "explanations.json").exists() and "--refresh-report" not in sys.argv:
        raise FileExistsError("Do not replace an existing completed interpretation")
    comparisons = build_comparisons(citations)
    rows = []
    for key, d in audit["decisions"].items():
        f, w = d["configuration"]["formation"], d["input"]["wind_direction"]
        mechanism, ref_ids = interpretation(d)
        alternatives = comparisons["by_key"][key]
        nearest = alternatives[0]
        contrast, contrast_refs = geometry_contrast(f, nearest["comparator_formation"], w)
        ref_ids = [r["id"] for r in refs if r["id"] in set(ref_ids + contrast_refs)]
        drop_a, drop_b = nearest["selected_swarm_drop_pp"], nearest["comparator_swarm_drop_pp"]
        relation = "lower" if drop_a < drop_b else "higher"
        cross_summary = (
            f"Compared with the best available {nearest['comparator_configuration']}, the selected profile has a "
            f"summed five-drone Bideal SOC drop of {drop_a:.3f} versus {drop_b:.3f} pp over 25 s "
            f"({abs(drop_a-drop_b)/drop_b*100:.1f}% {relation}). "
            + ("The time advantage accompanies lower aggregate discharge. " if drop_a < drop_b else
               "The time advantage comes from the charging workload distribution, not lower aggregate discharge. ")
            + contrast
        )
        if nearest["selected_spacing_cm"] != nearest["comparator_spacing_cm"]:
            cross_summary += " This compares complete configurations; the companion matched-spacing table holds spacing fixed."
        d["cross_formation_comparisons"] = alternatives
        d["cross_formation_explanation_en"] = cross_summary
        d["aerodynamic_interpretation_en"] = mechanism
        d["aerodynamic_evidence_status"] = "Literature-informed hypothesis, not demonstrated in these flight records"
        d["reference_ids"] = ref_ids
        d["references"] = [r for r in refs if r["id"] in ref_ids]
        d["explanation_en"] = d["data_explanation_en"] + " " + mechanism + " " + cross_summary + " " + " ".join(citations[r] for r in ref_ids)
        s = d["sensitivity"]
        k, strength = d["input"]["charging_pad_availability"], d["input"]["wind_strength"]
        rows.append(dict(key=key, charging_pads=k, wind_direction=w, wind_strength=strength,
                         condition=f"{strength.title()} {w}wind", configuration=d["paper_configuration"],
                         total_time_min=round(d["total_time_minutes"], 3), gap_s=round(d["gap_to_runner_up_seconds"], 3),
                         runner_up=d["runner_up_configuration"], min_trials=min(d["position_run_counts"]),
                         resampling_frequency=s["nominal_winner_selection_frequency"],
                         valid_draws=s["bootstrap_valid"], invalid_draws=s["bootstrap_invalid"],
                         loo_retained=s["loo_retained"], loo_evaluable=s["loo_evaluable"],
                         explanation_en=d["explanation_en"], explanation_zh=d["data_explanation_zh"],
                         caveats=" ".join(d["caveats"])))
    out = dict(audit, decisions=audit["decisions"], reviewed_literature=refs,
               cross_formation_summary=comparisons["summary"])
    (OUT / "explanations.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n")
    pd.DataFrame(rows).to_csv(OUT / "decision_explanations.csv", index=False)
    # A BibTeX companion preserves source identity; no unverified DOI/volume/pages.
    entries = []
    for r in refs:
        fields = dict(title="{"+r["title"]+"}", author=r["authors"], year=r["year"],
                      howpublished=r["venue"], url=r["url"])
        if r.get("doi"):
            fields["doi"] = r["doi"]
        entries.append("@misc{" + r["key"] + ",\n" + ",\n".join(f"  {k} = {{{v}}}" for k,v in fields.items()) + "\n}")
    (OUT / "references.bib").write_text("\n\n".join(entries)+"\n")
    sql = (OUT / "decision_margins.sql").read_text()
    connection = sqlite3.connect(":memory:")
    pd.read_csv(BASE / "candidate_rankings.csv").to_sql("candidate_rankings", connection, index=False)
    sql_results = pd.read_sql_query(sql, connection)
    connection.close()
    sensitivity_sql = (OUT / "sensitivity_summary.sql").read_text()
    connection = sqlite3.connect(":memory:")
    pd.read_csv(OUT / "resampling_sensitivity.csv").to_sql("resampling_sensitivity",connection,index=False)
    deletion_data = pd.read_csv(OUT / "leave_one_trial_out.csv")
    for column in ("evaluable", "retained"):
        deletion_data[column] = deletion_data[column].fillna(False).astype(bool).astype(int)
    deletion_data.to_sql("leave_one_trial_out",connection,index=False)
    sensitivity_results = pd.read_sql_query(sensitivity_sql,connection).set_index("key")
    connection.close()
    for row in rows:
        selected = sql_results[(sql_results.charging_pads == row["charging_pads"]) & (sql_results.wind_direction == row["wind_direction"]) & (sql_results.wind_strength == row["wind_strength"])].iloc[0]
        assert selected.configuration == row["configuration"]
        assert abs(selected.gap_s-row["gap_s"]) < .00051
        assert abs(selected.total_time_min-row["total_time_min"]) < .00051
        # Preserve extra audit columns while using the independently queried values.
        row.update(gap_s=float(selected.gap_s), total_time_min=float(selected.total_time_min))
        assert sensitivity_results.loc[row["key"], "valid_deletions"] == row["loo_evaluable"]
        assert sensitivity_results.loc[row["key"], "retained_deletions"] == row["loo_retained"]
    local_sources = [dict(id="decisions", label="Frozen Medium configuration decisions and candidate profiles",
                          path=str((BASE / "decision_function.json").relative_to(ROOT)),
                          query=dict(engine="SQLite in memory; upstream Python / NumPy charging audit", language="sql", sql=sql, executed_at=generated,
                                     description="SQL independently ranks all 275 frozen candidate scores and computes the 30 winning margins. Upstream discharge normalization and charging schedules are audited in interpret_medium_decisions.py; no real records or winners change.",
                                     tables_used=[str((BASE / f).relative_to(ROOT)) for f in ("decision_function.json", "preprocessed_75pct_250cm.csv", "candidate_rankings.csv")],
                                     filters=["Real 250 cm forward-flight Medium data only; no wind-tunnel or synthetic flights", "Fixed preprocessed Bideal SOC 75%; duration 25 s; identical available pads", "Four safety exclusions; one further cell lacks real data"],
                                     metric_definitions={"total_time_min":"(25 s + minimum parallel charging makespan)/60", "gap_s":"Second-smallest minus smallest computed total time, seconds", "min_trials":"Smallest usable real-run count among the five positions of the selected configuration"})),
                     dict(id="sensitivity", label="Whole-trial resampling and leave-one-trial-out audit",
                          path=str((OUT / "resampling_sensitivity.csv").relative_to(ROOT)),
                          query=dict(engine="SQLite in memory; upstream Python / NumPy resampling", language="sql",sql=sensitivity_sql,executed_at=generated, description="SQL recomputes deletion denominators and retention from the deletion-level audit and joins reviewed bootstrap results. Reproducible companion output_py/interpret_medium_decisions.py; 2,000 whole-trial bootstrap attempts per wind condition, seed 20260909.",
                                     tables_used=["analysis_results/medium_forward_bideal_v3_with_b15_20260909/run_drone_rates.csv",str((OUT / "leave_one_trial_out.csv").relative_to(ROOT))],
                                     metric_definitions={"resampling_frequency":"Fraction of evaluable bootstrap draws in which the nominal winner remains a minimum; not a posterior probability", "valid_draws":"Draws retaining at least one usable observation for every candidate-position; incomplete draws are excluded and counted", "loo_retained":"Evaluable whole-flight deletions, within the wind condition, preserving the nominal winner"}))]
    pub_sources = [dict(id=r["id"], label=f"{r['title']} ({r['year']})", href=r["url"]) for r in refs]
    for mode in ("best_available_spacing", "matched_spacing"):
        local_sources.append(dict(id="comparison_"+mode, label="Cross-formation comparison: "+mode.replace("_", " "),
            path=str((OUT/f"cross_formation_{mode}.csv").relative_to(ROOT)),
            query=dict(engine="SQLite in memory", language="sql", sql=(OUT/f"cross_formation_{mode}.sql").read_text(),
                       executed_at=generated, description="Executed join of frozen optimizer scores and five-position Medium rates. "
                       "The winner is compared with each other formation at its best available spacing, or at matching spacing.",
                       tables_used=[str((BASE/f).relative_to(ROOT)) for f in ("candidate_rankings.csv", "preprocessed_75pct_250cm.csv")],
                       filters=["Same wind direction, wind level and available charging-pad count", "Different formation", "Observed non-excluded profiles only"],
                       metric_definitions={"gap_s":"Comparator minus selected total time in seconds", "selected_swarm_drop_pp":"Sum of five selected Bideal position rates times 25/60; percentage points",
                                           "comparator_swarm_drop_pp":"Sum of five comparator Bideal position rates times 25/60; percentage points", "rate_profile":"P1-P5 Bideal percentage points per minute"})))
    sources = local_sources + pub_sources
    title = "Why each Medium configuration is selected"
    blocks, tables, charts = [], [], []
    def md(id, body, source_id=None):
        b = dict(id=id, type="markdown", body=body)
        if source_id:
            b["sourceId"] = source_id
        blocks.append(b)
    def table(id, title, dataset, columns, sort, source_id="decisions"):
        tables.append(dict(id=id, title=title, dataset=dataset, sourceId=source_id,
                           columns=[dict(field=f,label=l,type="text" if kind=="text" else "number") for f,l,kind in columns],
                           defaultSort=dict(field=sort,direction="asc")))
        blocks.append(dict(id=id+"_block",type="table",tableId=id))
    md("title", "# "+title)
    md("summary", "## Technical summary\n\n"
       "All **30 decisions** now have a short explanation linking the measured discharge profile to the charging bottleneck. "
       "The selected configurations are unchanged. In this paper-facing version, **low wind** and **high wind** replace the two fan-level labels.\n\n"
       "The explanation follows two connected steps: formation and wind alter rotor inflow, aerodynamic loads and motor demand; "
       "the resulting measured discharge profile determines the charging workload. Changing the number of pads changes the scheduling bottleneck, "
       "not the aerodynamics of the same flight. Detailed statistical checks are grouped in the method notes after the decisions.","decisions")
    md("baseline", "## The comparison uses a fixed Medium baseline\n\n"
       "The source cohort contains **134 real forward-flight runs** and **630 usable drone–run Medium records**, aggregated into **55 formation–spacing–wind profiles**, each with five position rates. "
       "Four unsafe combinations remain excluded; low sidewind column 50 cm is also unavailable because it lacks an eligible real profile. "
       "Neither wind-tunnel simulations nor new synthetic flights enter this analysis.\n\n"
       "SOC translation is **preprocessing**, outside the function: each equivalent Bideal battery starts at 75%, travels 250 cm in 25 s, "
       "and arrives at 75 − rate × 25/60. Rates use Bideal percentage points per minute. The previously frozen, battery-specific Medium selection and normalization are retained. "
       "This is not a claim that every physical battery starts at raw 75%, or that the original flights had identical durations.\n\n"
       "Total time means the fixed 25 s segment plus the time until the last drone finishes charging. Charging uses the existing exponential model "
       "with 0→99% taking 90 min, identical pads immediately available on arrival, no preemption and no discharge while waiting. "
       "These are calculation assumptions, not charging measurements from the flight experiment. There is no configuration-switch penalty in this baseline.")
    md("labels", "## Low and high describe wind, not battery stages\n\n"
       "Use **low wind (former Lv1)** and **high wind (former Lv2)**. These are categorical settings of the present experiment, "
       "not Beaufort classes or measured wind speeds. No value in m/s is inferred. Keep raw filenames and historical labels unchanged for traceability. "
       "Use *Medium SOC* when discussing the battery stage, to avoid confusing it with wind strength. "
       "Formation spelling in the paper is **echelon**; the historical code key `echalon` remains compatible.")
    md("map", "## Charging availability changes what makes a configuration preferable\n\n"
       "Read each row across the five charging-pad counts. Several wind conditions select different configurations as charging becomes more parallel. "
       "These are exact lookups among the observed, non-excluded candidates—not decisions for arbitrary unequal SOCs or a globally optimal multi-segment mission.","decisions")
    map_rows=[]
    for w,l in [(w,l) for w in ("head","side","tail") for l in (1,2)]:
        ds=[out["decisions"][f"{k}|{w}|{l}"] for k in range(1,6)]
        map_rows.append(dict(condition=f"{'Low' if l==1 else 'High'} {w}wind", **{f"k{k}":ds[k-1]["paper_configuration"] for k in range(1,6)}))
    table("map","Selected configuration", "map", [("condition","Wind","text")]+[(f"k{k}",f"{k} pad{'s' if k>1 else ''}","text") for k in range(1,6)],"condition")
    md("map_figure", "## Publication-ready decision map\n\n"
       "A separate paper figure reproduces all 30 cells above with wind direction and low/high wind on the rows and available charging pads on the columns. "
       "Each cell directly labels formation and spacing, so colour is not needed to interpret it. "
       "The companion is supplied as a vector PDF, editable SVG and 600 dpi PNG, with an English caption and a two-column LaTeX figure snippet. "
       "The figure preserves the frozen decisions and does not display a learned unequal-SOC policy.", "decisions")
    md("math", "## The direct explanation is a charging-workload calculation\n\n"
       "For each candidate c, let rᵢ(c) be its empirical Medium rate and sᵢ = 75 − 25rᵢ/60 its preprocessed arrival SOC. "
       "Its individual charging time is qᵢ = [5400/ln(100)] ln(100−sᵢ), in seconds, for the 99% target. "
       "The function is **F(k, direction, strength) = argmin_c [25 + min_partition max_pad Σ qᵢ]**. "
       "SOC alignment is not a fourth function input.\n\n"
       "In these profiles every qᵢ lies between 3,774.81 and 4,206.17 s. Sorting them as q(1) ≤ … ≤ q(5) makes the bottleneck particularly transparent:\n\n"
       "- **1 pad:** q(1)+q(2)+q(3)+q(4)+q(5).\n"
       "- **2 pads:** q(1)+q(2)+q(3); the two longer jobs share the other pad.\n"
       "- **3 pads:** max[q(1)+q(4), q(2)+q(3)]; q(5) is charged alone.\n"
       "- **4 pads:** q(1)+q(2); the other three jobs have separate pads.\n"
       "- **5 pads:** q(5).\n\n"
       "All **275** candidate scores were checked against exhaustive set-partition scheduling. "
       "The middle three expressions are valid for these near-equal charging jobs, not a universal shortcut for arbitrary SOCs. "
       "In particular, uniform consumption is not always the best schedule when pads are scarce.","decisions")
    md("literature", "## Rotorcraft research explains the physical links\n\n"
       "Single-vehicle tests justify considering wind, attitude and propulsion together [R1]. "
       "Power need not increase monotonically with airspeed [R2], and a small-quadrotor fan experiment found a comparatively flat power response [R3]. "
       "The useful physical distinction is between the thrust needed to resist external loading and the power needed to generate that thrust in the local inflow.\n\n"
       "Proximity-flight work supports dependence on relative separation [R4]; multi-vehicle interactions can be non-additive [R5]. "
       "Flow measurements show that wake structure changes with flight regime [R6]. These justify candidate-specific empirical rates, "
       "because separation and wake location affect the load on each aircraft.\n\n"
       "The related formation-selection study [R7] used a Phantom model and much faster simulated flight; its drag-only and combined-force rankings differed, "
       "and it did not test tailwind. Its rankings therefore cannot replace the present Tello observations. "
       "The five-Tello demonstration [R8] is an architectural precedent. Fixed-wing V-formation results [R9] are not a rotorcraft energy law."
       .replace("[R1]",citations["R1"]).replace("[R2]",citations["R2"]).replace("[R3]",citations["R3"])
       .replace("[R4]",citations["R4"]).replace("[R5]",citations["R5"]).replace("[R6]",citations["R6"])
       .replace("[R7]",citations["R7"]).replace("[R8]",citations["R8"]).replace("[R9]",citations["R9"]))
    for section_id, heading, body in mechanism_sections(citations):
        md(section_id, "## " + heading + "\n\n" + body)
    md("geometry", "## Read aerodynamic roles relative to the airflow\n\n"
       "P1–P5 below are the positions in the frozen forward-flight table, not a reconstruction from the newer wind-tunnel layout. "
       "Upstream and downstream are defined relative to the local airflow, so reversing the wind can exchange the aerodynamic roles of the same numbered positions. "
       "The physical discussion applies rotorcraft mechanisms to the stated geometry; the selection itself is calculated from measured discharge rates, "
       "which combine propulsion, position corrections and the battery-gauge response. The flight data do not separately resolve these power components.")
    datasets=dict(map=map_rows,decisions=rows,preview=rows[:10])
    extra_notes={
        ("head",1): "Front 75 cm has rates [1.64, 10.83, 0.97, 2.47, 7.34] pp/min. P1, P3 and P4 are especially low, helping the 1–4 pad objectives. "
                    "With five pads, echelon 75 cm reduces the largest rate from 10.83 to 8.40 pp/min. That is why the answer changes without any change of wind.",
        ("head",2): "Front 50 cm has low P3 and P5 rates but a larger P4 rate; echelon 75 cm has a smaller maximum rate. "
                    "The intermediate-pad outcomes follow the short-job combinations, rather than one formation dominating every position.",
        ("side",1): "Front 50 cm minimizes the 1–3 pad objectives, whereas diamond 75 cm has the best short pair for four pads. "
                    "Column 75 cm only narrowly improves the maximum job for five pads. The unobserved column 50 cm cell prevents a complete safe-candidate comparison.",
        ("side",2): "Front 50 cm has a relatively compact rate range of 5.40–6.46 pp/min, helping its worst-job objective. "
                    "Echelon 50 cm has a very low P3 rate (1.62 pp/min), which helps short-job groups for two and four pads. "
                    "The one-pad margin between these formations is only 3.04 s.",
        ("tail",1): "Vee 50 cm has a small calculated P2 charging requirement, while its P4 demand is larger. "
                    "This helps the short-job groupings for limited charging pads. With five pads, diamond 75 cm instead wins by limiting the largest charging requirement.",
        ("tail",2): "Column 50 cm has low P3–P5 rates but its P1 rate is higher (12.58 pp/min). "
                    "The low-rate positions reduce its combined charging workload, so it wins the 1–4 pad calculations. "
                    "For five pads, echelon 50 cm wins by just 0.68 s over diamond 75 cm."
    }
    for wind,level in [(w,l) for w in ("head","side","tail") for l in (1,2)]:
        strength="low" if level==1 else "high"
        id=f"{wind}_{strength}"
        subset=[r for r in rows if r["wind_direction"]==wind and r["wind_strength"]==strength]
        datasets[id]=subset
        md(id+"_intro",f"## {strength.title()} {wind}wind: the rate profile and charging bottleneck\n\n"+extra_notes[(wind,level)],"decisions")
        table(id,"Computed decisions and margins",id,[("charging_pads","Pads","number"),("configuration","Configuration","text"),
             ("total_time_min","Total min","number"),("runner_up","Runner-up","text"),("gap_s","Gap s","number")],"charging_pads")
        for k in range(1,6):
            d=out["decisions"][f"{k}|{wind}|{level}"]
            md(id+f"_{k}_reason",f"### {k} charging pad{'s' if k>1 else ''}: {d['paper_configuration']}\n\n"+d["explanation_en"])
    cs = comparisons["summary"]
    md("cross_formation_intro", "## Compare the selected configuration with all four other formations\n\n"
       f"The following **{cs['best_other_formation_comparisons']} comparisons** hold wind direction, wind level and charging-pad count constant. "
       "Each alternative formation uses its own best available spacing; the next table additionally holds spacing fixed. "
       "The selected configuration has lower summed normalized discharge in **112** comparisons, but wins despite higher summed discharge in **8**. "
       "These are descriptive comparisons of the same frozen profiles reused across pad counts, not 120 independent experiments.\n\n"
       "For example, in low headwind with five pads, echelon 75 cm consumes a summed 14.601 Bideal pp over 25 s versus front 75 cm's 9.687 pp, "
       "yet finishes 41.02 s earlier: its largest position rate is 8.395 rather than 10.830 pp/min. "
       "This is a reduction in the limiting charging job, not a claim that echelon uses less aggregate battery charge. "
       "With one pad under the same wind, front 75 cm instead wins.\n\n"
       "Per-position vectors and all four short comparative explanations for each decision are retained in the comparison datasets and the cross-formation companion. "
       "Summed Bideal SOC drop is a normalized consumption measure, not a measured energy value in joules; P1-P5 refer to each formation's recorded positions.", "comparison_best_available_spacing")
    for mode, heading in (("best_available_spacing", "Other formations at their best available spacing"),
                          ("matched_spacing", "Other formations at matching spacing")):
        dataset="cross_formation_"+mode
        datasets[dataset]=[dict(r, reference_ids=", ".join(r["reference_ids"])) for r in comparisons[mode]]
        table(dataset, heading, dataset,
              [("condition","Wind","text"),("charging_pads","Pads","number"),
               ("selected_configuration","Selected","text"),("comparator_configuration","Comparator","text"),
               ("selected_swarm_drop_pp","Selected drop pp","number"),("comparator_swarm_drop_pp","Comparator drop pp","number"),
               ("gap_s","Extra time s","number"),("selected_rate_profile","Selected P1-P5 rates","text"),
               ("comparator_rate_profile","Comparator P1-P5 rates","text"),("selected_critical_pad","Selected last pad","text"),
               ("comparator_critical_pad","Comparator last pad","text"),("explanation_en","Comparative explanation","text")],
              "condition", "comparison_"+mode)
    md("uncertainty", "## Method notes: trial-deletion check and decision margins\n\n"
       "Whole-flight leave-one-out checks can change the selected configuration in **25 of the 30 states**. "
       "This check describes sensitivity to which trials are averaged; it is separate from the physical explanation of each configuration. "
       "The following chart compares the five-pad winner–runner-up time differences. Five states across all pad counts have gaps below 10 s.")
    chart_rows=[r for r in rows if r["charging_pads"]==5]
    datasets["five_pad_gaps"]=chart_rows
    charts.append(dict(id="gaps",title="Five-pad decision margins",subtitle="Gap to the second-ranked configuration; seconds; fixed Medium baseline",
                       showDescription=True,type="bar",intent="comparison",dataset="five_pad_gaps",sourceId="decisions",layout="full",
                       encodings=dict(x=dict(field="condition",type="nominal",label="Wind condition"),
                                      y=dict(field="gap_s",type="quantitative",label="Margin (s)")),
                       settings=dict(orientation="vertical",sort="none",showValues=True),palette=dict(kind="sequential",root="blue"),
                       legend=dict(show=False),labels=dict(values="all"),valueFormat="number"))
    blocks.append(dict(id="gaps_block",type="chart",chartId="gaps"))
    md("bootstrap", "## Method notes: whole-trial resampling\n\n"
       "Within each candidate cell, the bootstrap resamples whole trials jointly across all five positions, preserving within-flight dependence. "
       "There are 2,000 attempted draws per wind condition. Draws missing any candidate-position rate are not ranked; "
       "evaluable counts range from 1,162 to 1,863. Results are conditional on this complete-profile requirement. "
       "The table reports how often the original choice was retained in the evaluable draws. "
       "Sparse cells, especially singleton positions, cannot reveal their full variability through resampling. Calibration and spatial-wind uncertainty are not propagated.","sensitivity")
    datasets["stability"]=[dict(r,retention=f"{r['resampling_frequency']:.1%}",loo=f"{r['loo_retained']}/{r['loo_evaluable']}") for r in rows]
    table("stability","Decision stability audit","stability",[("condition","Wind","text"),("charging_pads","Pads","number"),
          ("configuration","Nominal winner","text"),("retention","Bootstrap retained","text"),("valid_draws","Valid draws","number"),
          ("loo","LOO retained / valid","text"),("min_trials","Min trials","number")],"condition","sensitivity")
    md("checks", "## Alternative charging assumptions did not change these 30 winners\n\n"
       "Three separate sensitivity checks retained all 30 configurations: constant-current charging to 99%, and the same exponential charging model "
       "to 80% or 90%. Their absolute times are different; these checks do not validate the original 90-minute charging assumption. "
       "Trial variability remains the larger warning in this particular audit.\n\n"
       "Across the 25 formation–wind pairs with both spacings observed, 75 cm has the lower average rate in 12 pairs and the lower maximum rate in 12 pairs. "
       "This descriptive count does not support a universal monotonic spacing rule; the pairs are not independent causal replications. "
       "The audit retains the source table rather than changing rates to match a preferred aerodynamic story.")
    md("next", "## Present the mechanism first, then the charging consequence\n\n"
       "For the current Medium analysis, explain the geometry, rotor inflow or drag-compensation mechanism, then identify the positions that set the charging completion time. "
       "Use the empirical rates for the numerical comparison and keep the trial checks in the method notes. "
       "A change caused solely by charging-pad availability is a scheduling effect; it does not require a different aerodynamic story. "
       "No additional flights are required to use this descriptive analysis. The unresolved issues are the empirical charging curve, "
       "residual measurement bias, run-specific airflow geometry and whether the same ranking transfers to unequal SOCs or another flight regime. "
       "Those questions are not silently answered by the present function.")
    md("refs", "## Reviewed research and transfer limits\n\n"+"\n\n".join(
        f"**{r['id']}.** {r['authors'].replace(' and ', ', ')} ({r['year']}). [{r['title']}]({r['url']}). {r['venue']}. "
        f"*Use:* {r['supports']} *Limit:* {r['boundary']}" for r in refs))
    artifact=dict(surface="report",manifest=dict(version=1,surface="report",title=title,generatedAt=generated,
                   blocks=blocks,tables=tables,charts=charts,sources=sources),
                  snapshot=dict(version=1,status="ready",generatedAt=generated,datasets=datasets),sources=sources)
    (OUT / "artifact.json").write_text(json.dumps(artifact,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    # Copy-ready companion: method and all concise explanations with bibliography.
    paper=["# Medium configuration decisions: thesis wording", "",
           "Wind strength is reported as low (historical Lv1) or high (historical Lv2). The analysis uses the preprocessed real forward-flight Medium baseline, not simulated wind-tunnel data.","",
           "## Method paragraph", "",
           "We calculate a position-specific Medium discharge rate for each measured wind direction, wind strength, formation and inter-drone spacing. "
           "Before applying the decision function, we normalize the battery differences using Bideal and translate the Medium SOC trajectories to a common 75% starting level. "
           "For a 250 cm segment lasting 25 s, these rates determine each drone's arrival SOC. Given the available charging pads, we calculate the minimum time for all five drones to complete charging under the stated charging model. "
           "The selected configuration minimizes the segment flight time plus this charging completion time among the observed, non-excluded configurations. "
           "We interpret the position-specific discharge profiles through rotor inflow, aerodynamic loading and position-holding effort, then explain how these profiles determine the charging workload.", "",
           "## Reading assumptions", "",
           "The charging target is 99%, with an assumed exponential curve taking 90 minutes from 0 to 99%. Total times are calculated, not directly observed. "
           "The candidate rates are linear only within Medium. P1–P5 are data positions, not inferred windward/leader labels. "
           "The physical discussion applies mechanisms from rotorcraft research to the formation geometry; the measured discharge rates determine the numerical choices. "
           "Detailed trial checks are collected separately in method_notes.md.", ""]
    for _, heading, body in mechanism_sections(citations):
        paper += ["## " + heading, "", body, ""]
    paper += ["## How the cross-formation comparison is made", "",
              "For each decision, we compare the selected configuration with the best available spacing of each other formation under the same wind and charging availability. "
              "We also make matched-spacing comparisons wherever both profiles are available. The five position rates, summed normalized SOC drop and critical charging workload are considered separately. "
              "Lower total time is not automatically described as lower aggregate discharge. The complete 120 comparisons are in cross_formation_comparisons.md; "
              "the paragraphs below highlight the closest alternative formation. The analysis uses recorded position labels without assuming that P1 has the same aerodynamic role in every formation.", ""]
    for wind,level in [(w,l) for w in ("head","side","tail") for l in (1,2)]:
        paper += [f"## {'Low' if level==1 else 'High'} {wind}wind", "", extra_notes[(wind,level)], ""]
        for k in range(1,6):
            d=out["decisions"][f"{k}|{wind}|{level}"]
            paper += [f"### {k} available charging pad{'s' if k>1 else ''}", "",d["explanation_en"],""]
    paper += ["## References", ""]+[f"{r['id']}. {r['authors'].replace(' and ', ', ')} ({r['year']}). [{r['title']}]({r['url']}). {r['venue']}." for r in refs]
    (OUT / "thesis_explanations.md").write_text("\n".join(paper)+"\n")
    method_notes = ["# Supporting method notes", "", "These checks accompany the mechanism-focused thesis text; the numerical audit is unchanged.", ""]
    for key, d in out["decisions"].items():
        method_notes += [f"## {d['paper_wind']}, {d['input']['charging_pad_availability']} pad(s): {d['paper_configuration']}", ""]
        method_notes += ["- " + note for note in d["caveats"]] + [""]
    (OUT / "method_notes.md").write_text("\n".join(method_notes)+"\n")
    notes=dict(audience="technical",delivery="mcp-app; HTML only after renderer failure",snapshot=generated,
               scope="Current 30 Medium decisions; no Overleaf edits, original data edits, flight changes or new ML training",
               required_structure_mapping=dict(title="title",technical_summary="summary",definitions="baseline/labels",method="math",
                   findings="map and six wind sections",uncertainty="uncertainty/bootstrap/checks",next_steps="next",further_questions="Unresolved issues in next; no new user questions needed"),
               chart_contract=dict(question="How large are the five-pad decision margins?",family="Comparison & Ranking",variant="single-series vertical bars",
                   grain="six wind conditions, k=5",fields=["condition","gap_s"],sufficiency="Six categorical comparisons, not a time trend",
                   palette="single blue root plus neutrals",non_color="category labels and numeric value labels",footprint="full report width",
                   source="Frozen decision table; contextual configuration, runner-up, trial count and resampling fields retained"),
               omitted_other_charts="All 30 decisions require exact lookup and prose; six repeated charts would duplicate the tables rather than explain the schedule.",
               source_limitations="R1 abstract/project record only; R9 abstract/introduction only; all other cited sources full text inspected; no paywall circumvention",
               sensitivity="Whole-trial bootstrap and deletion, calibration fixed; incomplete draws explicitly counted; not a formal significance claim",
               geometry="No new wind-tunnel layout is assigned to historical forward-flight positions",
               revision="Adds comparisons against all other formations at best and matched spacings; all 30 choices, numerical evidence and existing charts/tables retained. Paper decision map exported separately as explicitly requested.",
               cross_formation_comparisons=comparisons["summary"],
               publication_figure="output/pdf/medium_configuration_decision_map.pdf; SVG, PNG, caption and LaTeX companion under output/figures",
               artifact_companion="thesis_explanations.md is copy-ready text, not a second report rendering mode")
    (OUT / "report_plan.json").write_text(json.dumps(notes,ensure_ascii=False,indent=2)+"\n")
    manifest=dict(version=1,source_sha256={**audit["source_sha256"],"raw_run_rates":audit["raw_rate_sha256"]},
                  output_sha256={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name not in {"manifest.json","validation_receipt.json","render_receipt.json"}},
                  validation=summary,source_kind="real forward-flight rates; literature-informed interpretation; no invented measurements")
    (OUT / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(decisions=len(rows),references=len(refs),blocks=len(blocks),artifact=str(OUT/"artifact.json")),indent=2))


if __name__ == "__main__":
    main()
