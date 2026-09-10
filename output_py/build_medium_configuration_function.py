"""Preprocess the real forward-flight Medium table, score, validate, freeze F.

Run from the repository root: python3 -m output_py.build_medium_configuration_function
No raw data edits, simulated inputs, ML training, flight control or app changes.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

from ml_policy.charging_model import (
    FULLY_CHARGED_SOC, ZERO_TO_FULLY_CHARGED_MINUTES, exponential_charging_minutes,
)
from ml_policy.oracle_optimizer import UNSAFE_STRUCTURES_BY_CONDITION, _optimal_parallel_charging_schedule


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis_results/medium_forward_bideal_v3_with_b15_20260909"
DEST = ROOT / "analysis_results/medium_configuration_function_v1_20260909"
SAFETY_SOURCE = ROOT / "frozen_data/discharge_rates/medium_bideal_v1_20260908/manifest.json"
START_SOC = 75.0  # Preprocessing only; not an input to the runtime function.
DISTANCE_CM = 250
FLIGHT_SECONDS = 25.0
TIE_TOLERANCE_SECONDS = 1e-7
POSITIONS = {f"drone_{i}": i for i in range(1, 6)}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cell(row):
    formation = row["formation"].replace("echelon", "echalon")
    return row["wind_direction"], int(row["wind_level"]), formation, int(row["inter_drone_spacing_cm"])


def brute_force_makespan(jobs_seconds, pads):
    """Independent validator: examine EVERY pad assignment, including symmetric ones."""
    return min(max(sum(jobs_seconds[i] for i in range(5) if assignment[i] == p)
                   for p in range(pads))
               for assignment in itertools.product(range(pads), repeat=5))


def preprocess():
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    if source_manifest["source_segment_cm"] != DISTANCE_CM or "not wind tunnel" not in source_manifest["scope"]:
        raise ValueError("Expected the frozen real 250 cm forward-flight cohort")
    medium_floor, medium_ceiling = source_manifest["Bideal_medium_soc_range"]
    if not medium_floor < START_SOC <= medium_ceiling:
        raise ValueError("Reference SOC is outside the Bideal middle band")
    model_path = ROOT / "analysis_results/battery_normalization_v3_with_b15_20260909/model.json"
    if digest(model_path) != source_manifest["normalization_model_sha256"]:
        raise ValueError("Battery normalization model differs from the source rate table")
    unsafe = {tuple(c) for c in json.loads(SAFETY_SOURCE.read_text())["unsafe_condition_cells"]}
    unsafe_in_solver = {(w, level, f, spacing) for (w, level), structures in UNSAFE_STRUCTURES_BY_CONDITION.items()
                        for f, spacing in structures}
    if unsafe != unsafe_in_solver:
        raise ValueError("Frozen and solver safety masks disagree")
    raw_details = read_csv(SOURCE / "run_drone_rates.csv")
    all_rates = read_csv(SOURCE / "discharge_rates_wide.csv")
    counts = {cell(r): r for r in read_csv(SOURCE / "sample_counts_wide.csv")}
    coverage = {(cell(r), int(r["position"])): r for r in read_csv(SOURCE / "coverage.csv")}
    seen, preprocessed = set(), []
    for row in all_rates:
        key = cell(row)
        if key in seen:
            raise ValueError(f"Duplicate rate row: {key}")
        seen.add(key)
        if key in unsafe:
            continue
        base = dict(wind_direction=key[0], wind_level=key[1], formation=key[2], inter_drone_spacing_cm=key[3],
                    reference_start_soc_Bideal=START_SOC, distance_cm=DISTANCE_CM, forward_duration_s=FLIGHT_SECONDS,
                    data_kind="derived_from_real_medium_forward_rates_not_new_flight")
        for position in range(1, 6):
            rate = float(row[f"position_{position}_discharge_rate_Bideal_pp_per_min"])
            if not math.isfinite(rate) or rate < 0:
                raise ValueError(f"Invalid discharge rate: {key}/{position}")
            run_values = [float(r["discharge_rate_Bideal_pp_per_min"]) for r in raw_details
                          if cell(r) == key and int(r["position"]) == position and r["status"] == "included"]
            count = int(counts[key][f"position_{position}_run_count"])
            if not run_values or len(run_values) != count or not math.isclose(rate, math.fsum(run_values) / count, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Rate/mean/count reconciliation failed: {key}/{position}")
            drop = rate * FLIGHT_SECONDS / 60
            arrival = START_SOC - drop
            if arrival < medium_floor:
                raise ValueError(f"Standardized trajectory crosses into Low: {key}/{position}")
            charge_seconds = 60 * exponential_charging_minutes(arrival)
            # Independent closed-form check of the shared charging implementation.
            tau = ZERO_TO_FULLY_CHARGED_MINUTES / math.log(100 / (100 - FULLY_CHARGED_SOC))
            independent_charge = 60 * tau * math.log((100 - arrival) / (100 - FULLY_CHARGED_SOC))
            if not math.isclose(charge_seconds, independent_charge, rel_tol=1e-12):
                raise ValueError("Charging calculation discrepancy")
            cov = coverage[(key, position)]
            for name, value in dict(discharge_rate_Bideal_pp_min=rate, drop_Bideal_pp=drop,
                    arrival_soc_Bideal=arrival, charging_seconds=charge_seconds,
                    run_count=count, rate_std_pp_min=cov["rate_std"],
                    short_trace_count=int(cov["short_trace_count"]), flat_trace_count=int(cov["flat_trace_count"])).items():
                base[f"position_{position}_{name}"] = value
        preprocessed.append(base)
    expected = set(itertools.product(("head", "side", "tail"), (1, 2), ("front", "column", "vee", "echalon", "diamond"), (50, 75)))
    missing = sorted(expected - unsafe - seen)
    return sorted(preprocessed, key=cell), source_manifest, sorted(unsafe), missing


def score(preprocessed):
    scores = []
    for row in preprocessed:
        for pads in range(1, 6):
            jobs = tuple(row[f"position_{i}_charging_seconds"] for i in range(1, 6))
            schedule = _optimal_parallel_charging_schedule(tuple(v / 60 for v in jobs), pads)
            charging = schedule.makespan_minutes * 60
            independent = brute_force_makespan(jobs, pads)
            if not math.isclose(charging, independent, abs_tol=1e-7):
                raise ValueError("Independent exhaustive charging validation failed")
            groups = [[f"drone_{i+1}" for i in indices] for indices in schedule.drone_indices_by_pad]
            if sorted(i for group in groups for i in group) != sorted(POSITIONS):
                raise ValueError("A drone is missing or duplicated in the charging schedule")
            loads = [v * 60 for v in schedule.pad_loads_minutes]
            if not math.isclose(sum(jobs), sum(loads), abs_tol=1e-7):
                raise ValueError("Charging work was not conserved")
            scores.append(dict(charging_pad_availability=pads, wind_direction=row["wind_direction"], wind_level=row["wind_level"],
                formation=row["formation"], inter_drone_spacing_cm=row["inter_drone_spacing_cm"],
                flight_seconds=FLIGHT_SECONDS, charging_completion_seconds=charging,
                total_time_seconds=FLIGHT_SECONDS + charging, total_time_minutes=(FLIGHT_SECONDS + charging) / 60,
                arrival_soc_by_position=[row[f"position_{i}_arrival_soc_Bideal"] for i in range(1, 6)],
                charging_seconds_by_position=list(jobs), charging_pad_groups=groups, charging_pad_loads_seconds=loads,
                min_position_run_count=min(row[f"position_{i}_run_count"] for i in range(1, 6)),
                short_trace_position_records=sum(row[f"position_{i}_short_trace_count"] for i in range(1, 6)),
                flat_trace_position_records=sum(row[f"position_{i}_flat_trace_count"] for i in range(1, 6))))
    return scores


def make_decisions(scores, missing):
    decisions, ranked = {}, []
    for pads, wind, level in itertools.product(range(1, 6), ("head", "side", "tail"), (1, 2)):
        candidates = sorted([dict(r) for r in scores if (r["charging_pad_availability"], r["wind_direction"], r["wind_level"]) == (pads, wind, level)],
                            key=lambda r: (r["total_time_seconds"], r["formation"], r["inter_drone_spacing_cm"]))
        if not candidates:
            raise ValueError(f"No eligible real candidates for {wind}/{level}")
        minimum = candidates[0]["total_time_seconds"]
        ties = [r for r in candidates if r["total_time_seconds"] - minimum <= TIE_TOLERANCE_SECONDS]
        best = min(ties, key=lambda r: (r["formation"], r["inter_drone_spacing_cm"]))
        runner = min([r for r in candidates if r is not best], key=lambda r: r["total_time_seconds"])
        missing_here = [dict(formation=f, inter_drone_spacing_cm=s) for w, l, f, s in missing if (w, l) == (wind, level)]
        key = f"{pads}|{wind}|{level}"
        decisions[key] = dict(
            input=dict(charging_pad_availability=pads, wind_direction=wind, wind_level=level),
            configuration=dict(formation=best["formation"], inter_drone_spacing_cm=best["inter_drone_spacing_cm"], position_assignment=POSITIONS.copy()),
            flight_seconds=FLIGHT_SECONDS, charging_completion_seconds=best["charging_completion_seconds"],
            total_time_seconds=best["total_time_seconds"], total_time_minutes=best["total_time_minutes"],
            arrival_soc_by_position=best["arrival_soc_by_position"], charging_seconds_by_position=best["charging_seconds_by_position"],
            charging_pad_groups=best["charging_pad_groups"], charging_pad_loads_seconds=best["charging_pad_loads_seconds"],
            candidate_count=len(candidates), runner_up=dict(formation=runner["formation"], inter_drone_spacing_cm=runner["inter_drone_spacing_cm"], total_time_seconds=runner["total_time_seconds"]),
            gap_to_runner_up_seconds=max(0, runner["total_time_seconds"] - best["total_time_seconds"]),
            tied_configurations=[dict(formation=r["formation"], inter_drone_spacing_cm=r["inter_drone_spacing_cm"]) for r in ties],
            missing_safe_configurations=missing_here, min_position_run_count=best["min_position_run_count"],
            short_trace_position_records=best["short_trace_position_records"], flat_trace_position_records=best["flat_trace_position_records"],
            interpretation="computed optimum among observed non-excluded configurations under fixed preprocessed baseline; statistical superiority not established")
        for rank, candidate in enumerate(candidates, 1):
            candidate.update(rank=rank, gap_to_best_seconds=candidate["total_time_seconds"] - minimum)
            ranked.append(candidate)
    return decisions, ranked


def csv_flat(row):
    return {k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()}


def build():
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite an existing analysis: {DEST}")
    paths = [SOURCE / filename for filename in ("manifest.json", "discharge_rates_wide.csv", "sample_counts_wide.csv", "run_drone_rates.csv", "coverage.csv")]
    paths += [SAFETY_SOURCE, ROOT / "analysis_results/battery_normalization_v3_with_b15_20260909/model.json",
              ROOT / "ml_policy/charging_model.py", ROOT / "ml_policy/oracle_optimizer.py", Path(__file__),
              ROOT / "ml_policy/medium_configuration_function.py"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    preprocessed, source_manifest, unsafe, missing = preprocess()
    scores = score(preprocessed)
    decisions, ranked = make_decisions(scores, missing)
    for path, expected_hash in before.items():
        if digest(ROOT / path) != expected_hash:
            raise ValueError(f"Source changed during calculation: {path}")
    assumptions = dict(reference_soc_preprocessing=75, Bideal_middle_band=source_manifest["Bideal_medium_soc_range"],
        flight_distance_cm=DISTANCE_CM, flight_duration_seconds=FLIGHT_SECONDS,
        batteries="five equivalent Bideal batteries; not five physical batteries at merely equal displayed percentages",
        position_assignment="identity retained; permutations are time-equivalent under equal SOC and identical charging assumptions",
        charging_model="existing exponential model", charging_target_soc=FULLY_CHARGED_SOC,
        zero_to_target_minutes=ZERO_TO_FULLY_CHARGED_MINUTES,
        charging_pads="1..5 identical pads, all available on arrival, one drone per pad at a time, non-preemptive charging",
        charging_completion="minimum time until every drone has completed charging; do not sum individual waiting times again",
        ground_wait="no battery drain while waiting on the ground", switching="not included; configuration is chosen before this segment",
        objective="single standardized flight segment + subsequent whole-swarm charging completion; not a global multi-segment optimum",
        evidence="unchanged equal-run mean Medium discharge rates; source short, partial, flat, and low-count warnings retained; no new filters",
        source_coverage="observed real forward-flight candidates only; no simulated or invented missing configurations",
        total_time="computed from empirical discharge rates and an assumed charging model, not directly measured total mission time")
    model = dict(schema_version=1, kind="medium_configuration_lookup", version=DEST.name,
                 function_inputs=["charging_pad_availability", "wind_direction", "wind_level"],
                 assumptions=assumptions, decisions=decisions)
    decision_rows = []
    for result in decisions.values():
        decision_rows.append(dict(**result["input"], formation=result["configuration"]["formation"],
            inter_drone_spacing_cm=result["configuration"]["inter_drone_spacing_cm"],
            total_time_seconds=result["total_time_seconds"], total_time_minutes=result["total_time_minutes"],
            charging_completion_seconds=result["charging_completion_seconds"],
            runner_up_formation=result["runner_up"]["formation"], runner_up_spacing_cm=result["runner_up"]["inter_drone_spacing_cm"],
            gap_to_runner_up_seconds=result["gap_to_runner_up_seconds"], candidate_count=result["candidate_count"],
            missing_safe_configurations=json.dumps(result["missing_safe_configurations"]),
            min_position_run_count=result["min_position_run_count"]))
    DEST.mkdir(parents=True)
    write_csv(DEST / "preprocessed_75pct_250cm.csv", preprocessed)
    write_csv(DEST / "candidate_rankings.csv", [csv_flat(r) for r in ranked])
    write_csv(DEST / "decision_table.csv", decision_rows)
    (DEST / "decision_function.json").write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n")
    audit = dict(source_version=source_manifest["version"], source_summary=source_manifest["summary"],
        preprocessed_configuration_count=len(preprocessed), candidate_evaluations=len(scores), input_states=len(decisions),
        unsafe_excluded=unsafe, missing_safe_configurations=missing,
        preprocessing="Bideal rate * 25/60 -> normalized drop; 75 - drop -> arrival; source SOC/timestamps untouched",
        assumptions=assumptions, source_sha256=before,
        verification=dict(all_mean_rates_and_position_counts_reconciled=True,
            all_reference_arrivals_in_Bideal_middle_band=True, all_schedules_independently_exhaustively_verified=True,
            each_drone_assigned_once=True, all_charging_job_loads_conserved=True,
            charging_formula_independently_verified=True, input_sources_unchanged=True),
        output_sha256={p.name: digest(p) for p in DEST.iterdir() if p.is_file()})
    (DEST / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = ["# Medium configuration function — fixed preprocessed baseline", "",
        "`F(charging_pad_availability, wind_direction, wind_level) -> configuration`", "",
        "SOC alignment is preprocessing, NOT a runtime input or operation. This is an exact finite lookup, not a neural network.", "",
        f"Source: {SOURCE.relative_to(ROOT)}. {len(preprocessed)} real-data configuration rows, {len(scores)} scored candidates across 30 input states.", "",
        "## Decision map", "", "Times below are minutes for the 25-second flight plus whole-swarm charging completion.", "",
        "| Wind | Level | 1 pad | 2 pads | 3 pads | 4 pads | 5 pads |", "|---|---:|---|---|---|---|---|"]
    for wind, level in itertools.product(("head", "side", "tail"), (1, 2)):
        entries = []
        for k in range(1, 6):
            r = decisions[f"{k}|{wind}|{level}"]; c = r["configuration"]
            entries.append(f"{c['formation']} {c['inter_drone_spacing_cm']} cm / {r['total_time_minutes']:.3f} min")
        lines.append(f"| {wind} | {level} | " + " | ".join(entries) + " |")
    lines += ["", "## Assumptions and limits", ""] + [f"- **{k}**: {v}" for k, v in assumptions.items()]
    lines += ["", f"- Explicit collision exclusions: {unsafe}", f"- Missing safe real-data configurations: {missing}",
        "- Equal-run averaging occurs before nonlinear charging evaluation. This is a standardized mean-rate comparison, not the mean of measured trial charging times.",
        "- A numerical winner, especially a tiny gap or low-count/flat-trace candidate, is not proof of statistically significant superiority.",
        "- No forward experiment can be replaced by a wind-tunnel simulation in this function.", "",
        "## Use", "", "```python", "from ml_policy.medium_configuration_function import select_medium_configuration",
        "configuration = select_medium_configuration(2, 'head', 1)", "```", "",
        "Use `explain_medium_configuration` with the same three inputs for exact times, runner-up gap, individual SOCs and the charging schedule.",
        "For reproducibility, the manifest records the source hashes and independent exhaustive-scheduling checks. Rankings retain every evaluated candidate."]
    (DEST / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ("preprocessed_configuration_count", "candidate_evaluations", "input_states", "missing_safe_configurations", "verification")}, indent=2))
    print("\n".join(lines[10:19]))


if __name__ == "__main__":
    build()
